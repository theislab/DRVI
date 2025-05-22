import logging
from collections.abc import Callable, Sequence

import numpy as np
import scvi
import torch
from anndata import AnnData
from torch.nn import functional as F

from drvi.scvi_tools_based.module._constants import MODULE_KEYS

logger = logging.getLogger(__name__)


class GenerativeMixin:
    """Mixins to interpret generative part of the method."""

    @torch.inference_mode()
    def iterate_on_decoded_latent_samples(
        self,
        z: np.ndarray,
        step_func: Callable,
        aggregation_func: Callable,
        lib: np.ndarray | None = None,
        cat_key: np.ndarray | None = None,
        cont_key: np.ndarray | None = None,
        batch_size=scvi.settings.batch_size,
    ) -> np.ndarray:
        r"""Iterate over decoder outputs and aggregate the results.

        Parameters
        ----------
        z
            Latent samples.
        step_func
            Function to apply to the decoder output at each step.
            generative_outputs and a store variable are given to the function
        aggregation_func
            Function to aggregate the step results.
        lib
            Library size array.
        cat_key
            Categorical covariates (required if model has categorical key).
        cont_key
            Continuous covariates.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        store = []
        self.module.eval()

        with torch.no_grad():
            for i in np.arange(0, z.shape[0], batch_size):
                slice = np.arange(i, min(i + batch_size, z.shape[0]))
                z_tensor = torch.tensor(z[slice])
                if lib is None:
                    lib_tensor = torch.tensor([1e4] * slice.shape[0])
                else:
                    lib_tensor = torch.tensor(lib[slice])
                cat_tensor = torch.tensor(cat_key[slice]) if cat_key is not None else None
                batch_tensor = None

                gen_input = self.module._get_generative_input(
                    tensors={
                        scvi.REGISTRY_KEYS.BATCH_KEY: batch_tensor,
                        scvi.REGISTRY_KEYS.LABELS_KEY: None,
                        scvi.REGISTRY_KEYS.CONT_COVS_KEY: torch.log(lib_tensor).unsqueeze(-1)
                        if self.summary_stats.get("n_extra_continuous_covs", 0) == 1
                        else None,
                        scvi.REGISTRY_KEYS.CAT_COVS_KEY: cat_tensor,
                    },
                    inference_outputs={
                        MODULE_KEYS.Z_KEY: z_tensor,
                        MODULE_KEYS.LIBRARY_KEY: lib_tensor,
                        MODULE_KEYS.LIKELIHOOD_ADDITIONAL_PARAMS_KEY: {},
                    },
                )
                gen_output = self.module.generative(**gen_input)
                step_func(gen_output, store)
        result = aggregation_func(store)
        return result

    @torch.inference_mode()
    def decode_latent_samples(
        self,
        z: np.ndarray,
        lib: np.ndarray | None = None,
        cat_key: np.ndarray | None = None,
        cont_key: np.ndarray | None = None,
        batch_size=scvi.settings.batch_size,
    ) -> np.ndarray:
        r"""Return the distribution produces by the decoder for the given latent samples.

        This is typically considered as :math:`p(x \mid z)`.

        Parameters
        ----------
        z
            Latent samples.
        lib
            Library size array.
        cat_key
            Categorical covariates (required if model has categorical key).
        cont_key
            Continuous covariates.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Return the mean of the distribution or the full distribution.
        """
        step_func = lambda gen_output, store: store.append(gen_output[MODULE_KEYS.PX_PARAMS_KEY]["mean"].detach().cpu())
        aggregation_func = lambda store: torch.cat(store, dim=0).numpy(force=True)

        return self.iterate_on_decoded_latent_samples(
            z=z,
            step_func=step_func,
            aggregation_func=aggregation_func,
            lib=lib,
            cat_key=cat_key,
            cont_key=cont_key,
            batch_size=batch_size,
        )

    @torch.inference_mode()
    def iterate_on_ae_output(
        self,
        adata: AnnData,
        step_func: Callable,
        aggregation_func: Callable,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        deterministic: bool = False,
    ) -> np.ndarray:
        r"""Iterate over autoencoder outputs and aggregate the results.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If `None`, defaults to the AnnData object used to initialize the model.
        step_func
            Function to apply to the autoencoder output at each step.
            inference_outputs, generative_outputs, losses, and a store variable are given to the function
        aggregation_func
            Function to aggregate the step results.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        deterministic
            Makes model fully deterministic (e.g. no sampling in the bottleneck).
        """
        adata = self._validate_anndata(adata)
        data_loader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        store = []
        try:
            if deterministic:
                self.module.fully_deterministic = True
            for tensors in data_loader:
                loss_kwargs = {"kl_weight": 1}
                inference_outputs, generative_outputs, losses = self.module(tensors, loss_kwargs=loss_kwargs)
                step_func(inference_outputs, generative_outputs, losses, store)
        except Exception as e:
            self.module.fully_deterministic = False
            raise e
        finally:
            self.module.fully_deterministic = False

        return aggregation_func(store)

    @torch.inference_mode()
    def get_reconstruction_effect_of_each_split(
        self,
        adata: AnnData | None = None,
        add_to_counts: float = 1.0,
        aggregate_over_cells: bool = True,
        deterministic: bool = True,
        **kwargs,
    ) -> np.ndarray:
        r"""Return the effect of each split on the reconstructed per sample.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If `None`, defaults to the AnnData object used to initialize the model.
        add_to_counts
            Value to add to the counts before computing the logarithm.
        aggregate_over_cells
            Whether to aggregate the effect over cells and return a value per dimension.
        deterministic
            Makes model fully deterministic (e.g. no sampling in the bottleneck).
        kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.
        """

        def calculate_effect(inference_outputs, generative_outputs, losses, store):
            if self.module.split_aggregation == "logsumexp":
                log_mean_params = generative_outputs[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY][
                    "mean"
                ]  # n_samples x n_splits x n_genes
                log_mean_params = F.pad(
                    log_mean_params, (0, 0, 0, 1), value=np.log(add_to_counts)
                )  # n_samples x (n_splits + 1) x n_genes
                effect_share = -torch.log(1 - F.softmax(log_mean_params, dim=-2)[:, :-1, :]).sum(
                    dim=-1
                )  # n_samples x n_splits
            elif self.module.split_aggregation == "sum":
                effect_share = torch.abs(generative_outputs[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY]["mean"]).sum(
                    dim=-1
                )  # n_samples x n_splits
            else:
                raise NotImplementedError("Only logsumexp and sum aggregations are supported for now.")
            return store.append(effect_share.detach().cpu())

        def aggregate_effects(store):
            return torch.cat(store, dim=0).numpy(force=True)

        output = self.iterate_on_ae_output(
            adata=adata,
            step_func=calculate_effect,
            aggregation_func=aggregate_effects,
            deterministic=deterministic,
            **kwargs,
        )

        if aggregate_over_cells:
            output = output.sum(axis=0)

        return output

    # @torch.inference_mode()
    def get_max_effect_of_splits_within_distribution(
        self,
        adata: AnnData | None = None,
        add_to_counts: float = 1.0,
        deterministic: bool = True,
        **kwargs,
    ):
        r"""
        Return the max effect of each split on the reconstructed expression params for all genes.

        These values are empirical and inexact for de reasoning of dimensions.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If `None`, defaults to the AnnData object used to initialize the model.
        add_to_counts
            Value to add to the counts before computing the logarithm.
        deterministic
            Makes model fully deterministic (e.g. no sampling in the bottleneck).
        kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.

        Returns
        -------
        np.ndarray
            Max effect of each split on the reconstructed expression params for all genes


        -------
        Example usage
        -------

        import utils.plotting._interpretability        >>> effects = model.get_max_effect_of_splits_within_distribution(add_to_counts=0.1)
        >>> effect_data = (
        ...     pd.DataFrame(
        ...         effects,
        ...         columns=model.adata.var_names,
        ...         index=embed.var["title"],
        ...     )
        ...     .loc[embed.var.query("vanished == False").sort_values("order")["title"]]
        ...     .T
        ... )
        >>> plot_info = list(effect_data.to_dict(orient="series").items())
        >>> utils.plotting._interpretability._bar_plot_top_differential_vars(plot_info)
        >>> utils.plotting._interpretability._umap_of_relevant_genes(adata, embed, plot_info, dim_subset=["DR 1"])
        """

        def calculate_effect(inference_outputs, generative_outputs, losses, store):
            if self.module.split_aggregation == "logsumexp":
                log_mean_params = generative_outputs[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY][
                    "mean"
                ]  # n_samples x n_splits x n_genes
                log_mean_params = F.pad(
                    log_mean_params, (0, 0, 0, 1), value=np.log(add_to_counts)
                )  # n_samples x (n_splits + 1) x n_genes
                effect_share = -torch.log(1 - F.softmax(log_mean_params, dim=-2)[:, :-1, :])
            elif self.module.split_aggregation == "sum":
                effect_share = torch.abs(generative_outputs[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY]["mean"])
            else:
                raise NotImplementedError("Only logsumexp and sum aggregations are supported for now.")
            effect_share = effect_share.amax(dim=0).detach().cpu().numpy(force=True)
            if len(store) == 0:
                store.append(effect_share)
            else:
                store[0] = np.maximum(store[0], effect_share)

        def aggregate_effects(store):
            return store[0]

        output = self.iterate_on_ae_output(
            adata=adata,
            step_func=calculate_effect,
            aggregation_func=aggregate_effects,
            deterministic=deterministic,
            **kwargs,
        )

        return output
