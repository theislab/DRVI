import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from drvi.model import DRVI


class TestDRVIModel:
    n = 1_000
    g = 200
    c = 2
    b = 5

    def make_test_adata(self, is_sparse=True):
        N, G, C, B = self.n, self.g, self.c, self.b

        ct_list = np.random.choice(range(C), N)[:, np.newaxis]
        batch_list = np.random.choice(range(B), N)[:, np.newaxis]
        batch_list_2 = np.random.choice(range(B), N)[:, np.newaxis]
        ct_array = (np.indices((N, C))[1] == ct_list) + 0.0
        g_exp_array = np.random.randint(0, 2, [G, C])
        exp_indicator = ct_array @ g_exp_array.T
        g_mean_list = np.exp(np.random.random(G) * 10 - 5)[:, np.newaxis]

        exp_matrix = np.random.poisson(exp_indicator * g_mean_list.T).astype(np.float32)

        adata = ad.AnnData(
            X=sparse.csr_matrix(exp_matrix),
            obs=pd.DataFrame(
                {
                    "cell_type": [f"ct_{ct}" for ct in ct_list[:, 0]],
                    "batch": [f"batch_{bid}" for bid in batch_list[:, 0]],
                    "batch_2": [f"batch_{bid}" for bid in batch_list_2[:, 0]],
                },
                index=[f"cell_{i}" for i in range(N)],
            ),
            var=pd.DataFrame(
                {
                    "gene_mean": g_mean_list[:, 0],
                    "gene_active_signature": np.apply_along_axis(
                        lambda x: "".join(x), axis=1, arr=g_exp_array.astype(str)
                    ),
                },
                index=[f"gene_{i}" for i in range(G)],
            ),
        )

        adata.obs["total_counts"] = adata.X.sum(axis=1)
        adata.layers["counts"] = adata.X.copy()
        adata.layers["lognorm"] = np.log1p(adata.X)

        if not is_sparse:
            adata.X = adata.X.A
            for l in ["counts", "lognorm"]:
                adata.layers[l] = adata.layers[l].A

        return adata

    def _general_integration_test(self, adata, layer="lognorm", data_kwargs=None, **kwargs):
        is_count_data = layer == "counts"
        setup_anndata_default_params = dict(  # noqa: C408
            categorical_covariate_keys=["batch"],
            layer=layer,
            is_count_data=is_count_data,
        )
        default_args = dict(  # noqa: C408
            n_latent=32,
            encoder_dims=[128],
            decoder_dims=[128],
            gene_likelihood="normal",
            categorical_covariates=["batch"],
            decoder_reuse_weights="everywhere",
        )
        DRVI.setup_anndata(adata, **{**setup_anndata_default_params, **(data_kwargs or {})})
        model = DRVI(adata, **{**default_args, **kwargs})
        print(model.module)
        model.train(accelerator="cpu", max_epochs=10)
        latent = model.get_latent_representation(adata)
        assert latent.shape[0] == adata.n_obs

    def test_dimension_reduction_with_no_batch(self):
        adata = self.make_test_adata()
        self._general_integration_test(
            adata,
            categorical_covariates=[],
            data_kwargs=dict(categorical_covariate_keys=[]),  # noqa: C408
        )

    def test_simple_integration(self):
        adata = self.make_test_adata()
        self._general_integration_test(adata)

    def test_simple_integration_with_masking(self):
        adata = self.make_test_adata()
        self._general_integration_test(adata, fill_in_the_blanks_ratio=0.5)

    def test_simple_integration_latent_splitting(self):
        adata = self.make_test_adata()
        self._general_integration_test(adata, n_latent=32, n_split_latent=-1)
        self._general_integration_test(adata, n_latent=32, n_split_latent=8)
        self._general_integration_test(adata, n_latent=32, n_split_latent=8, split_method="power")
        self._general_integration_test(adata, n_latent=32, n_split_latent=8, split_method="split_map")
        self._general_integration_test(
            adata, n_latent=32, n_split_latent=8, split_method="split", split_aggregation="max"
        )

    def test_simple_integration_mean_activation(self):
        adata = self.make_test_adata()
        self._general_integration_test(adata, n_latent=32, n_split_latent=32, mean_activation="identity")
        self._general_integration_test(adata, n_latent=32, n_split_latent=32, mean_activation="relu")
        self._general_integration_test(adata, n_latent=32, n_split_latent=32, mean_activation="leaky_relu_0.4")
        self._general_integration_test(adata, n_latent=32, n_split_latent=32, mean_activation="elu_0.4")

    def test_decoder_reusing(self):
        adata = self.make_test_adata()
        for reuse_strategy in ["nowhere"]:
            self._general_integration_test(
                adata,
                n_latent=32,
                n_split_latent=8,
                split_method="split",
                split_aggregation="logsumexp",
                decoder_reuse_weights=reuse_strategy,
            )

    def test_integration_with_different_likelihoods(self):
        adata = self.make_test_adata()
        for gene_likelihood in ["nb", "normal", "pnb", "normal_sv"]:
            self._general_integration_test(
                adata,
                gene_likelihood=gene_likelihood,
                layer="counts" if gene_likelihood in ["nb", "pnb"] else "lognorm",
            )

    def test_integration_without_covariates(self):
        adata = self.make_test_adata()
        self._general_integration_test(
            adata,
            categorical_covariates=[],
            data_kwargs=dict(categorical_covariate_keys=[]),  # noqa: C408
        )

    def test_integration_with_different_covariate_modelings(self):
        adata = self.make_test_adata()
        for encode_covariates in [False, True]:
            for cms in [
                "one_hot",
                "emb_shared",
                "emb",
                "one_hot_linear",
                "emb_shared",
                "emb_linear",
            ]:
                self._general_integration_test(
                    adata, covariate_modeling_strategy=cms, encode_covariates=encode_covariates
                )

    def test_integration_with_different_var_activations(self):
        adata = self.make_test_adata()
        for var_activation in ["exp", "pow2"]:
            self._general_integration_test(adata, var_activation=var_activation)

    def test_integration_with_different_priors(self):
        adata = self.make_test_adata()
        for prior, prior_init_obs in [
            ("normal", None),
            ("gmm_5", None),
            ("gmm_5", adata.obs.index.to_series().sample(5)),
            ("vamp_5", adata.obs.index.to_series().sample(5)),
        ]:
            self._general_integration_test(adata, prior=prior, prior_init_obs=prior_init_obs)

    def test_multilevel_batch_integration(self):
        adata = self.make_test_adata()
        self._general_integration_test(
            adata,
            categorical_covariates=["batch", "batch_2"],
            data_kwargs=dict(categorical_covariate_keys=["batch", "batch_2"]),  # noqa: C408
        )
        self._general_integration_test(
            adata,
            categorical_covariates=["batch", "batch_2"],
            data_kwargs=dict(categorical_covariate_keys=["batch", "batch_2"]),  # noqa: C408
            prior="vamp_5",
            prior_init_obs=adata.obs.index.to_series().sample(5),
        )

    def _general_query_to_reference(self, adata_reference, adata_query, layer="lognorm", data_kwargs=None, **kwargs):
        is_count_data = layer == "counts"
        setup_anndata_default_params = dict(  # noqa: C408
            categorical_covariate_keys=["batch"],
            layer=layer,
            is_count_data=is_count_data,
        )
        default_args = dict(  # noqa: C408
            n_latent=32,
            encoder_dims=[128],
            decoder_dims=[128],
            gene_likelihood="normal",
            categorical_covariates=["batch"],
            decoder_reuse_weights="everywhere",
            encode_covariates=True,
        )
        DRVI.setup_anndata(adata_reference, **{**setup_anndata_default_params, **(data_kwargs or {})})
        model = DRVI(adata_reference, **{**default_args, **kwargs})
        model.train(accelerator="cpu", max_epochs=10)

        latent_reference = model.get_latent_representation(adata_reference)

        DRVI.prepare_query_anndata(adata_query, model)
        transfer_model = model.load_query_data(adata_query, model)
        transfer_model.train(accelerator="cpu", max_epochs=10, plan_kwargs={"lr": 0.1, "weight_decay": 0.0})
        latent_query = transfer_model.get_latent_representation(adata_query)
        transfer_model.train(accelerator="cpu", max_epochs=10, plan_kwargs={"lr": 0.1, "weight_decay": 0.0})
        latent_reference_after_train = transfer_model.get_latent_representation(adata_reference)
        latent_query_after_train = transfer_model.get_latent_representation(adata_query)

        assert np.sum((latent_reference - latent_reference_after_train) ** 2) < 1e-8
        assert np.sum((latent_query - latent_query_after_train) ** 2) > 1e-3
        assert latent_query_after_train.shape[0] == adata_query.n_obs

    def test_simple_query_to_reference_mapping(self):
        adata = self.make_test_adata()
        adata_reference = adata[adata.obs["batch"] != "batch_0"].copy()
        adata_query = adata[adata.obs["batch"] == "batch_0"].copy()
        self._general_query_to_reference(adata_reference, adata_query)

    def test_query_to_reference_mapping_with_different_cov_models(self):
        adata = self.make_test_adata()
        adata_reference = adata[adata.obs["batch"] != "batch_0"].copy()
        adata_query = adata[adata.obs["batch"] == "batch_0"].copy()
        for cms in [
            "one_hot",
            "emb_shared",
            "emb",
            "one_hot_linear",
            "emb_shared",
            "emb_linear",
        ]:
            self._general_query_to_reference(adata_reference, adata_query, covariate_modeling_strategy=cms)

    def test_query_to_reference_mapping_with_different_splits(self):
        adata = self.make_test_adata()
        adata_reference = adata[adata.obs["batch"] != "batch_0"].copy()
        adata_query = adata[adata.obs["batch"] == "batch_0"].copy()
        for n_split_latent in [1, 4, 32]:
            self._general_query_to_reference(adata_reference, adata_query, n_split_latent=n_split_latent)

    def test_query_to_reference_mapping_with_different_splits_and_different_cov_models(self):
        adata = self.make_test_adata()
        adata_reference = adata[adata.obs["batch"] != "batch_0"].copy()
        adata_query = adata[adata.obs["batch"] == "batch_0"].copy()
        for cms in ["one_hot", "emb_shared", "emb", "one_hot_linear", "emb_shared", "emb_linear"]:
            for n_split_latent in [1, 4, 32]:
                self._general_query_to_reference(
                    adata_reference, adata_query, n_split_latent=n_split_latent, covariate_modeling_strategy=cms
                )

    def test_continues_covariates(self):
        adata = self.make_test_adata()
        adata.obs["log_library_size_as_cont_cov"] = np.log1p(adata.layers["counts"].sum(-1))
        for encode_covariates in [False, True]:
            self._general_integration_test(
                adata,
                gene_likelihood="nb",
                layer="counts",
                encode_covariates=encode_covariates,
                data_kwargs=dict(continuous_covariate_keys=["log_library_size_as_cont_cov"]),  # noqa: C408
            )

    def test_ann_data_categorical_transfer_on_query_to_reference_mapping(self):
        # TODO: revise and complete this file
        adata = self.make_test_adata()
        adata_reference = adata[adata.obs["batch"] != "batch_0"].copy()
        adata_query = adata[adata.obs["batch"] == "batch_0"].copy()

        layer = "counts"
        is_count_data = layer == "counts"
        setup_anndata_default_params = dict(  # noqa: C408
            categorical_covariate_keys=["batch"],
            layer=layer,
            is_count_data=is_count_data,
        )
        default_args = dict(  # noqa: C408
            n_latent=32,
            encoder_dims=[128],
            decoder_dims=[128],
            gene_likelihood="normal",
            categorical_covariates=["batch"],
            decoder_reuse_weights="everywhere",
            encode_covariates=True,
        )
        DRVI.setup_anndata(adata_reference, **setup_anndata_default_params)
        model = DRVI(adata_reference, **default_args)
        model.train(accelerator="cpu", max_epochs=10)

        DRVI.prepare_query_anndata(adata_query, model)
        transfer_model = model.load_query_data(adata_query, model)
        transfer_model.train(accelerator="cpu", max_epochs=10, plan_kwargs={"lr": 0.1, "weight_decay": 0.0})

        for current_adata in (adata_query, adata_reference, adata):
            print("*****")
            current_adata = transfer_model._validate_anndata(current_adata)
            scdl = transfer_model._make_data_loader(
                adata=current_adata,
                indices=None,
                batch_size=128,
            )
            for tensors in scdl:
                print(tensors)
                inference_inputs = transfer_model.module._get_inference_input(tensors)
                print(inference_inputs)
                transfer_model.module.inference(**inference_inputs)
                break
