from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from torch.distributions import Normal
from torch.utils.data import DataLoader

from drvi.nn_modules.embedding import MultiEmbedding
from drvi.nn_modules.layer.factory import LayerFactory
from drvi.nn_modules.noise_model import (
    LogNegativeBinomialNoiseModel,
    NegativeBinomialNoiseModel,
    NormalNoiseModel,
    PoissonNoiseModel,
)
from drvi.nn_modules.prior import GaussianMixtureModelPrior, StandardPrior, VampPrior
from drvi.scvi_tools_based.nn import DecoderDRVI, Encoder

TensorDict = dict[str, torch.Tensor]


class DRVIModule(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE :cite:p:`Lopez18`.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    n_split_latent
        Number of splits in the latent space. -1 means split all dimensions (n_split_latent=n_latent).
    split_aggregation
        How to aggregate splits in the last layer of the decoder
    split_method
        How to make splits.
        "split" for splitting the latent space,
        "power" for transforming the latent space to n_split vectors of size n_latent
        "split_map" for splitting the latent space then map each to latent space using unique transformations
    decoder_reuse_weights
        Were to reuse the weights of the decoder layers when using splitting
        Possible values are 'everywhere', 'last', 'intermediate', 'nowhere'. Defaults to "everywhere"/
    encoder_dims
        Number of nodes in hidden layers of the encoder.
    decoder_dims
        Number of nodes in hidden layers of the decoder.
    n_cats_per_cov
        Number of categories for each categorical covariate
    n_continuous_cov
        Number of continuous covariates
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` >= 1. The covariates are concatenated to the input of subsequent hidden layers.
    categorical_covariate_dims
        Emb dim of covariate keys if applicable
    covariate_modeling_strategy
        The strategy model takes to remove covariates
    use_batch_norm
        Whether to use batch norm in layers.
    affine_batch_norm
        Whether to use affine batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    input_dropout_rate
        Dropout rate to apply to the input
    encoder_dropout_rate
        Dropout rate to apply to each of the encoder hidden layers
    decoder_dropout_rate
        Dropout rate to apply to each of the decoder hidden layers
    gene_likelihood
        gene likelihood model
    prior
        Prior model. defaults to normal.
    prior_init_dataloader
        Dataloader constructed to initialize the prior (or maintain in vamp).
    var_activation
        The activation function to ensure positivity of the variatinal distribution. Defaults to "exp".
    mean_activation
        The activation function at the end of mean encoder. Defaults to "identity".
        Possible values are "identity", "relu", "leaky_relu", "leaky_relu_{slope}", "elu", "elu_{min_vaule}".
    encoder_layer_factory
        A layer Factory instance for build encoder layers
    decoder_layer_factory
        A layer Factory instance for build decoder layers
    extra_encoder_kwargs
        Extra keyword arguments passed into encoder.
    extra_decoder_kwargs
        Extra keyword arguments passed into decoder.
    """

    def __init__(
        self,
        n_input: int,
        n_latent: int = 32,
        n_split_latent: int | None = -1,
        split_aggregation: Literal["sum", "logsumexp", "max"] = "logsumexp",
        split_method: Literal["split", "power", "split_map"] = "split_map",
        decoder_reuse_weights: Literal["everywhere", "last", "intermediate", "nowhere"] = "everywhere",
        encoder_dims: Sequence[int] = (128, 128),
        decoder_dims: Sequence[int] = (128, 128),
        n_cats_per_cov: Iterable[int] | None = (),
        n_continuous_cov: int = 0,
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = False,
        categorical_covariate_dims: Sequence[int] = (),
        covariate_modeling_strategy: Literal[
            "one_hot",
            "emb",
            "emb_shared",
            "one_hot_linear",
            "emb_linear",
            "emb_shared_linear",
        ] = "one_hot",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        affine_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        fill_in_the_blanks_ratio: float = 0.0,
        input_dropout_rate: float = 0.0,
        encoder_dropout_rate: float = 0.1,
        decoder_dropout_rate: float = 0.0,
        gene_likelihood: Literal[
            "normal",
            "normal_v",
            "normal_sv",
            "poisson",
            "poisson_orig",
            "nb",
            "nb_sv",
            "nb_libnorm",
            "nb_loglib_rec",
            "nb_libnorm_loglib_rec",
            "nb_loglibnorm_all",
            "nb_orig",
            "nb_softmax",
            "nb_orig_libnorm",
            "pnb",
            "pnb_sv",
            "pnb_softmax",
        ] = "pnb_softmax",
        prior: Literal["normal", "gmm_x", "vamp_x"] = "normal",
        prior_init_dataloader: DataLoader | None = None,
        var_activation: Callable | Literal["exp", "pow2", "2sig"] | Callable = "exp",
        mean_activation: Callable | str = "identity",
        encoder_layer_factory: LayerFactory = None,
        decoder_layer_factory: LayerFactory = None,
        extra_encoder_kwargs: dict | None = None,
        extra_decoder_kwargs: dict | None = None,
    ):
        super().__init__()
        self.n_latent = n_latent
        if n_split_latent == -1:
            n_split_latent = n_latent
        self.n_split_latent = n_split_latent
        self.split_aggregation = split_aggregation
        self.latent_distribution = "normal"
        self.gene_likelihood = gene_likelihood

        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates

        self.gene_likelihood_module = self._construct_gene_likelihood_module(gene_likelihood)
        self.fill_in_the_blanks_ratio = fill_in_the_blanks_ratio

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        affine_batch_norm_encoder = affine_batch_norm == "encoder" or affine_batch_norm == "both"
        affine_batch_norm_decoder = affine_batch_norm == "decoder" or affine_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        assert covariate_modeling_strategy in [
            "one_hot",
            "emb",
            "emb_shared",
            "one_hot_linear",
            "emb_linear",
            "emb_shared_linear",
        ]
        if covariate_modeling_strategy in ["emb_shared", "emb_shared_linear"] and len(n_cats_per_cov) > 0:
            self.shared_covariate_emb = MultiEmbedding(
                n_cats_per_cov, categorical_covariate_dims, init_method="normal", max_norm=1.0
            )
        else:
            self.register_module("shared_covariate_emb", None)

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            layers_dim=encoder_dims,
            input_dropout_rate=input_dropout_rate,
            dropout_rate=encoder_dropout_rate,
            n_cat_list=n_cats_per_cov if self.encode_covariates else [],
            n_continuous_cov=n_continuous_cov if self.encode_covariates else 0,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            affine_batch_norm=affine_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            mean_activation=mean_activation,
            layer_factory=encoder_layer_factory,
            covariate_modeling_strategy=covariate_modeling_strategy,
            categorical_covariate_dims=categorical_covariate_dims if self.encode_covariates else [],
            **(extra_encoder_kwargs or {}),
        )
        self.decoder = DecoderDRVI(
            n_latent,
            n_input,
            n_split=n_split_latent,
            split_aggregation=split_aggregation,
            split_method=split_method,
            reuse_weights=decoder_reuse_weights,
            gene_likelihood_module=self.gene_likelihood_module,
            layers_dim=decoder_dims,
            dropout_rate=decoder_dropout_rate,
            n_cat_list=n_cats_per_cov,
            n_continuous_cov=n_continuous_cov,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            affine_batch_norm=affine_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            layer_factory=decoder_layer_factory,
            covariate_modeling_strategy=covariate_modeling_strategy,
            categorical_covariate_dims=categorical_covariate_dims,
            **(extra_decoder_kwargs or {}),
        )

        self.prior = self._construct_prior(prior, prior_init_dataloader)
        self.fully_deterministic = False

    def _construct_gene_likelihood_module(self, gene_likelihood):
        if gene_likelihood == "normal":
            return NormalNoiseModel(model_var="fixed=1")
        elif gene_likelihood == "normal_v":
            return NormalNoiseModel(model_var="dynamic")
        elif gene_likelihood == "normal_sv":
            return NormalNoiseModel(model_var="feature")
        elif gene_likelihood == "poisson":
            return PoissonNoiseModel(mean_transformation="exp", library_normalization="none")
        elif gene_likelihood == "poisson_orig":
            return PoissonNoiseModel(mean_transformation="softmax", library_normalization="none")
        elif gene_likelihood in ["nb", "nb_sv"]:
            return NegativeBinomialNoiseModel(dispersion="feature", library_normalization="none")
        elif gene_likelihood in ["nb_libnorm"]:
            return NegativeBinomialNoiseModel(dispersion="feature", library_normalization="x_lib")
        elif gene_likelihood in ["nb_loglib_rec"]:
            return NegativeBinomialNoiseModel(dispersion="feature", library_normalization="x_loglib")
        elif gene_likelihood in ["nb_libnorm_loglib_rec"]:
            return NegativeBinomialNoiseModel(dispersion="feature", library_normalization="div_lib_x_loglib")
        elif gene_likelihood in ["nb_loglibnorm_all"]:
            return NegativeBinomialNoiseModel(dispersion="feature", library_normalization="x_loglib_all")
        elif gene_likelihood in ["nb_orig", "nb_softmax"]:
            return NegativeBinomialNoiseModel(
                dispersion="feature", mean_transformation="softmax", library_normalization="none"
            )
        elif gene_likelihood == "nb_orig_libnorm":
            return NegativeBinomialNoiseModel(
                dispersion="feature", mean_transformation="softmax", library_normalization="x_lib"
            )
        elif gene_likelihood in ["pnb", "pnb_sv"]:
            return LogNegativeBinomialNoiseModel(dispersion="feature", library_normalization="none")
        elif gene_likelihood in ["pnb_softmax"]:
            return LogNegativeBinomialNoiseModel(
                dispersion="feature", mean_transformation="softmax", library_normalization="none"
            )
        else:
            raise NotImplementedError()

    def _construct_prior(self, prior, prior_init_dataloader=None):
        if prior == "normal":
            return StandardPrior()
        elif prior.startswith("gmm_"):
            n_components = int(prior.split("_")[1])
            if prior_init_dataloader is not None:
                inference_output = self.inference(**self._get_inference_input(next(iter(prior_init_dataloader))))
                init_data = inference_output["qz_m"], inference_output["qz_v"]
            else:
                init_data = None
            return GaussianMixtureModelPrior(n_components, self.n_latent, data=init_data)
        elif prior.startswith("vamp_"):
            n_components = int(prior.split("_")[1])
            if prior_init_dataloader is not None:

                def preparation_function(prepared_input):
                    x = prepared_input["encoder_input"]
                    args = []
                    kwargs = {"cat_full_tensor": prepared_input["cat_full_tensor"]}
                    return x, args, kwargs

                model_input = self._input_pre_processing(**self._get_inference_input(next(iter(prior_init_dataloader))))
            else:
                raise ValueError("VaMP prior needs input samples as pseudo-inputs.")
            return VampPrior(
                n_components,
                self.z_encoder,
                model_input,
                input_type="scvi",
                trainable_keys=("encoder_input",),
                fixed_keys=("cat_full_tensor",),
                preparation_function=preparation_function,
            )
        else:
            raise NotImplementedError()

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        x = tensors[REGISTRY_KEYS.X_KEY]

        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)

        input_dict = {"x": x, "cont_covs": cont_covs, "cat_covs": cat_covs}
        return input_dict

    def _input_pre_processing(self, x, cont_covs=None, cat_covs=None):
        # log the input to the variational distribution for numerical stability
        x_, gene_likelihood_additional_info = self.gene_likelihood_module.initial_transformation(x)
        # Note: this is different from scvi implementation of library size that is log transformed
        # All our noise models accept non-normalized library to work
        library = x.sum(1)

        encoder_input = x_

        return {
            "encoder_input": encoder_input,
            "cat_full_tensor": cat_covs if self.encode_covariates else None,
            "cont_full_tensor": cont_covs if self.encode_covariates else None,
            "library": library,
            "gene_likelihood_additional_info": gene_likelihood_additional_info,
        }

    @auto_move_data
    def inference(self, x, cont_covs=None, cat_covs=None):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        pre_processed_input = self._input_pre_processing(x, cont_covs, cat_covs).copy()
        x_ = pre_processed_input["encoder_input"]

        # Mask if needed
        if self.fill_in_the_blanks_ratio > 0.0 and self.training:
            assert cont_covs is None  # We do not consider cont_cov here
            x_mask = torch.where(torch.rand_like(x_) >= self.fill_in_the_blanks_ratio, 1.0, 0.0)
            x_ = x_ * x_mask
            # TODO: check this:
            # x_ = x_ * x_mask / x_mask.mean(dim=1, keepdim=True)
        else:
            x_mask = None

        # Prepare shared emb
        if self.shared_covariate_emb is not None and self.encode_covariates:
            pre_processed_input["cat_full_tensor"] = self.shared_covariate_emb(
                pre_processed_input["cat_full_tensor"].int()
            )

        # get variational parameters via the encoder networks
        qz_m, qz_v, z = self.z_encoder(
            x_,
            cat_full_tensor=pre_processed_input["cat_full_tensor"],
            cont_full_tensor=pre_processed_input["cont_full_tensor"],
        )

        outputs = {
            "z": z,
            "qz_m": qz_m,
            "qz_v": qz_v,
            "library": pre_processed_input["library"],
            "x_mask": x_mask,
            "gene_likelihood_additional_info": pre_processed_input["gene_likelihood_additional_info"],
        }
        return outputs

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        if self.fully_deterministic:
            z = inference_outputs["qz_m"]
        library = inference_outputs["library"]
        gene_likelihood_additional_info = inference_outputs["gene_likelihood_additional_info"]

        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)

        input_dict = {
            "z": z,
            "library": library,
            "gene_likelihood_additional_info": gene_likelihood_additional_info,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    @auto_move_data
    def generative(self, z, library, gene_likelihood_additional_info, cont_covs=None, cat_covs=None):
        """Runs the generative model."""
        if self.shared_covariate_emb is not None:
            cat_covs = self.shared_covariate_emb(cat_covs.int())
        # form the likelihood
        px, params, original_params = self.decoder(
            z,
            cat_full_tensor=cat_covs,
            cont_full_tensor=cont_covs,
            library=library,
            gene_likelihood_additional_info=gene_likelihood_additional_info,
        )

        return {
            "px": px,
            "params": params,
            "original_params": original_params,
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Loss function."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        x_mask = inference_outputs["x_mask"]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        px = generative_outputs["px"]

        kl_divergence_z = self.prior.kl(Normal(qz_m, torch.sqrt(qz_v))).sum(dim=1)
        if self.fill_in_the_blanks_ratio > 0.0 and self.training:
            reconst_loss = -(px.log_prob(x) * (1 - x_mask)).sum(dim=-1)
        else:
            reconst_loss = -px.log_prob(x).sum(dim=-1)
        # For MSE this should be equivalent (in terms of backward gradients) to:
        # reconst_loss = torch.nn.GaussianNLLLoss(reduction='none')(x, px.loc, px.scale ** 2).sum(dim=-1)
        assert kl_divergence_z.shape == reconst_loss.shape

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = 0.0

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = {"kl_divergence_z": kl_divergence_z.sum()}
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics={
                "mse": torch.nn.functional.mse_loss(x, px.mean, reduction="none").sum(dim=1).mean(dim=0),
            },
        )

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> torch.Tensor:
        # Note: Not tested
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)  # noqa: C408
        (
            _,
            generative_outputs,
        ) = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        dist = generative_outputs["px"]

        if n_samples > 1:
            exprs = dist.sample().permute([1, 2, 0])  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors: TensorDict, n_mc_samples: int):
        # Note: Not tested
        """Marginal ll."""
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities

            p_z = Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)).log_prob(z).sum(dim=-1)
            p_x_zl = -reconst_loss

            to_sum[:, i] = p_z + p_x_zl

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl
