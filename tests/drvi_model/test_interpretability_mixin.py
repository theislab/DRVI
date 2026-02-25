import anndata as ad
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy import sparse

import drvi


class TestInterpretabilityMixin:
    """Minimal tests for InterpretabilityMixin public methods."""

    n = 100
    g = 50
    c = 2
    b = 4

    def _make_test_adata(self, is_sparse=True):
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
            adata.X = adata.X.toarray()
            for l in ["counts", "lognorm"]:
                adata.layers[l] = adata.layers[l].toarray()
        return adata

    def _setup_and_train_model(self, adata, n_latent=8, max_epochs=2, **kwargs):
        drvi.model.DRVI.setup_anndata(
            adata,
            categorical_covariate_keys=["batch"],
            layer="counts",
            is_count_data=True,
        )
        model_kwargs = dict(
            n_latent=n_latent,
            encoder_dims=[64],
            decoder_dims=[64],
            gene_likelihood="pnb",
            decoder_reuse_weights="everywhere",
        )
        model_kwargs.update(kwargs)
        model = drvi.model.DRVI(adata, **model_kwargs)
        model.train(accelerator="cpu", max_epochs=max_epochs)
        return model

    @pytest.fixture(scope="class")
    def trained_model_outputs(self):
        adata = self._make_test_adata()
        model = self._setup_and_train_model(adata)
        latent = model.get_latent_representation(adata)
        embed = ad.AnnData(latent, obs=adata.obs.copy())
        return adata, model, embed
    
    def test_get_reconstruction_effect_of_each_split(self, trained_model_outputs):
        adata, model, _embed = trained_model_outputs
        n_splits = model.module.n_split_latent

        effects = model.get_reconstruction_effect_of_each_split(adata=adata, aggregate_over_cells=True)
        assert effects.shape == (n_splits,)
        assert np.all(effects >= 0)

        cell_effects = model.get_reconstruction_effect_of_each_split(
            adata=adata, aggregate_over_cells=False
        )
        assert cell_effects.shape == (adata.n_obs, n_splits)
    
    def test_get_reconstruction_effect_directional(self, trained_model_outputs):
        adata, model, _embed = trained_model_outputs
        n_splits = model.module.n_split_latent

        effects = model.get_reconstruction_effect_of_each_split(
            adata=adata, aggregate_over_cells=True, directional=True
        )
        assert effects.shape == (2, n_splits)

    def test_set_latent_dimension_stats(self, trained_model_outputs):
        adata, model, embed = trained_model_outputs
        model.set_latent_dimension_stats(embed)
        for col in ["original_dim_id", "reconstruction_effect", "order", "min", "max", "title", "vanished"]:
            assert col in embed.var.columns, f"Missing column: {col}"
        assert embed.var.shape[0] == embed.X.shape[1]
        assert np.issubdtype(embed.var["reconstruction_effect"].dtype, np.floating)

    @pytest.fixture(scope="class")
    def trained_model_outputs_with_stats(self, trained_model_outputs):
        adata, model, embed = trained_model_outputs
        model.set_latent_dimension_stats(embed)
        return adata, model, embed

    def test_get_effect_of_splits_within_distribution(self, trained_model_outputs_with_stats):
        adata, model, _embed = trained_model_outputs_with_stats
        n_splits = model.module.n_split_latent
        n_genes = adata.n_vars

        result = model.get_effect_of_splits_within_distribution(
            adata=adata, aggregations="max", directional=True
        )
        assert "max" in result
        assert result["max"].shape == (2, n_splits, n_genes)

        result_nd = model.get_effect_of_splits_within_distribution(
            adata=adata, aggregations="max", directional=False
        )
        assert result_nd["max"].shape == (n_splits, n_genes)

    def test_get_effect_of_splits_out_of_distribution(self, trained_model_outputs_with_stats):
        adata, model, embed = trained_model_outputs_with_stats
        result = model.get_effect_of_splits_out_of_distribution(
            embed, n_steps=4, n_samples=2, directional=True
        )
        for key in ["min_possible", "max_possible", "combined"]:
            assert key in result
            assert result[key].shape == (2, embed.n_vars, adata.n_vars)

    def test_calculate_interpretability_scores_inplace(self, trained_model_outputs_with_stats):
        adata, model, embed = trained_model_outputs_with_stats
        embed_copy = embed.copy()
        out = model.calculate_interpretability_scores(
            embed_copy, methods="OOD", directional=True, inplace=True
        )
        assert out is None
        assert "OOD_combined_positive" in embed_copy.varm
        assert "OOD_combined_negative" in embed_copy.varm

    def test_calculate_interpretability_scores_return(self, trained_model_outputs_with_stats):
        adata, model, embed = trained_model_outputs_with_stats
        result = model.calculate_interpretability_scores(
            embed, methods="IND", directional=False, inplace=False
        )
        assert isinstance(result, dict)
        assert "IND_max" in result
        assert result["IND_max"].shape[0] == embed.n_vars
        assert result["IND_max"].shape[1] == adata.n_vars

    def test_get_interpretability_scores(self, trained_model_outputs_with_stats):
        adata, model, embed = trained_model_outputs_with_stats
        embed_copy = embed.copy()
        model.calculate_interpretability_scores(embed_copy, methods="OOD", inplace=True)

        df = model.get_interpretability_scores(
            embed_copy, adata, key="OOD_combined", directional=True
        )
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == adata.n_vars
        assert df.shape[1] == 2 * embed_copy.n_vars  # positive + negative per dimension

        model.calculate_interpretability_scores(
            embed_copy, methods="OOD", directional=False, inplace=True
        )
        df_nd = model.get_interpretability_scores(
            embed_copy, adata, key="OOD_combined", directional=False
        )
        assert df_nd.shape[0] == adata.n_vars
        assert df_nd.shape[1] == embed_copy.n_vars

    def test_plot_interpretability_scores(self, trained_model_outputs_with_stats):
        adata, model, embed = trained_model_outputs_with_stats
        embed_copy = embed.copy()
        model.calculate_interpretability_scores(embed_copy, methods="OOD", inplace=True)
        fig = model.plot_interpretability_scores(
            embed_copy, adata, show=False, score_threshold=-1.0
        )
        assert fig is not None
        plt.close(fig)
