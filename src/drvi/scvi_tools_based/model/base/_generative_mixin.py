import logging
from collections.abc import Callable, Sequence

import numpy as np
import scvi
import torch
from anndata import AnnData
from scvi import REGISTRY_KEYS
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GenerativeMixin:
    """Mixin class for generative model interpretation and analysis.

    This mixin provides methods for analyzing the generative part of variational
    autoencoder models. It includes utilities for decoding latent representations,
    iterating over model outputs, and analyzing the effects of different model
    components on the reconstruction.

    Notes
    -----
    This mixin is designed to work with scVI-based models that have a generative
    component. It provides high-level interfaces for:
    - Decoding latent samples to reconstruct data
    - Iterating over model outputs with custom functions
    - Analyzing reconstruction effects of different model splits
    - Computing maximum effects across the data distribution

    All methods operate in inference mode and handle batching automatically
    to manage memory usage with large datasets.
    """

    @torch.inference_mode()
    def iterate_on_decoded_latent_samples(
        self,
        z: np.ndarray,
        step_func: Callable,
        aggregation_func: Callable,
        lib: np.ndarray | None = None,
        cat_values: np.ndarray | None = None,
        cont_values: np.ndarray | None = None,
        batch_size=scvi.settings.batch_size,
        map_cat_values: bool = False,
    ) -> np.ndarray:
        """Iterate over decoder outputs and aggregate the results.

        This method processes latent samples through the generative model in batches,
        applies a custom function to each batch output, and aggregates the results.

        Parameters
        ----------
        z
            Latent samples with shape (n_samples, n_latent).
        step_func
            Function to apply to the decoder output at each step.
            Should accept (generative_outputs, store) as arguments.
        aggregation_func
            Function to aggregate the step results from the store.
            Should accept the store list and return the final result.
        lib
            Library size array with shape (n_samples,).
            If None, defaults to 1e4 for all samples.
        cat_values
            Categorical covariates with shape (n_samples, n_cat_covs).
            Required if model has categorical covariates.
        cont_values
            Continuous covariates with shape (n_samples, n_cont_covs).
        batch_size
            Minibatch size for data loading into model.
        map_cat_values
            Whether to map categorical covariates to integers based on
            the AnnData manager pipeline.

        Returns
        -------
        np.ndarray
            Aggregated results from processing all latent samples.

        Notes
        -----
        This method operates in inference mode and processes data in batches
        to manage memory usage. The step_func receives the generative outputs
        and a store variable for accumulating results.

        If map_cat_values is True, categorical values are automatically mapped
        to integers using the model's category mappings.

        Examples
        --------
        >>> import numpy as np
        >>> # Define custom step function to extract means
        >>> def extract_means(gen_output, store):
        ...     store.append(gen_output["params"]["mean"].detach().cpu())
        >>> # Define aggregation function to concatenate results
        >>> def concatenate_results(store):
        ...     return torch.cat(store, dim=0).numpy()
        >>> # Process latent samples
        >>> z = np.random.randn(50, 32)  # assuming 32 latent dimensions
        >>> result = model.iterate_on_decoded_latent_samples(
        ...     z=z, step_func=extract_means, aggregation_func=concatenate_results
        ... )
        >>> print(result.shape)  # (50, n_genes)
        """
        store = []
        self.module.eval()

        if cat_values is not None and map_cat_values:
            if cat_values.ndim == 1:  # For a user not noticing cat_values should be 2d!
                cat_values = cat_values.reshape(-1, 1)
            mapped_values = np.zeros_like(cat_values)
            for i, (_label, map_keys) in enumerate(
                self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["mappings"].items()
            ):
                cat_mapping = dict(zip(map_keys, range(len(map_keys)), strict=False))
                mapped_values[:, i] = np.vectorize(cat_mapping.get)(cat_values[:, i])
            cat_values = mapped_values.astype(np.int32)

        with torch.no_grad():
            for i in np.arange(0, z.shape[0], batch_size):
                slice = np.arange(i, min(i + batch_size, z.shape[0]))
                z_tensor = torch.tensor(z[slice])
                if lib is None:
                    lib_tensor = torch.tensor([1e4] * slice.shape[0])
                else:
                    lib_tensor = torch.tensor(lib[slice])
                cat_tensor = torch.tensor(cat_values[slice]) if cat_values is not None else None
                cont_tensor = torch.tensor(cont_values[slice]) if cont_values is not None else None
                batch_tensor = None

                gen_input = self.module._get_generative_input(
                    tensors={
                        REGISTRY_KEYS.BATCH_KEY: batch_tensor,
                        REGISTRY_KEYS.LABELS_KEY: None,
                        REGISTRY_KEYS.CONT_COVS_KEY: cont_tensor,
                        REGISTRY_KEYS.CAT_COVS_KEY: cat_tensor,
                    },
                    inference_outputs={
                        "z": z_tensor,
                        "library": lib_tensor,
                        "gene_likelihood_additional_info": {},
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
        cat_values: np.ndarray | None = None,
        cont_values: np.ndarray | None = None,
        batch_size=scvi.settings.batch_size,
        map_cat_values: bool = False,
    ) -> np.ndarray:
        r"""Return the distribution produced by the decoder for the given latent samples.

        This method computes :math:`p(x \mid z)`, the reconstruction distribution
        for given latent samples. It returns the mean of the reconstruction
        distribution for each sample.

        Parameters
        ----------
        z
            Latent samples with shape (n_samples, n_latent).
        lib
            Library size array with shape (n_samples,).
            If None, defaults to 1e4 for all samples.
        cat_values
            Categorical covariates with shape (n_samples, n_cat_covs).
            Required if model has categorical covariates.
        cont_values
            Continuous covariates with shape (n_samples, n_cont_covs).
        batch_size
            Minibatch size for data loading into model.
        map_cat_values
            Whether to map categorical covariates to integers based on
            the AnnData manager pipeline.

        Returns
        -------
        np.ndarray
            Reconstructed means with shape (n_samples, n_genes).

        Notes
        -----
        This method is equivalent to computing the expected value of the
        reconstruction distribution :math:`E[p(x \mid z)]`. It's useful for:
        - Generating synthetic data from latent samples
        - Analyzing model reconstructions
        - Visualizing the generative capabilities of the model

        Examples
        --------
        >>> import numpy as np
        >>> # Generate random latent samples
        >>> z = np.random.randn(100, 32)  # assuming 32 latent dimensions
        >>> # Decode to get reconstructed means
        >>> reconstructed = model.decode_latent_samples(z)
        >>> print(reconstructed.shape)  # (100, n_genes)
        >>> # With categorical covariates
        >>> cat_covs = np.array([0, 1, 0, 1] * 25)  # batch labels
        >>> reconstructed = model.decode_latent_samples(z, cat_values=cat_covs)
        """
        step_func = lambda gen_output, store: store.append(gen_output["params"]["mean"].detach().cpu())
        aggregation_func = lambda store: torch.cat(store, dim=0).numpy(force=True)

        return self.iterate_on_decoded_latent_samples(
            z=z,
            step_func=step_func,
            aggregation_func=aggregation_func,
            lib=lib,
            cat_values=cat_values,
            cont_values=cont_values,
            batch_size=batch_size,
            map_cat_values=map_cat_values,
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
        """Iterate over autoencoder outputs and aggregate the results.

        This method processes data through the full autoencoder (encoder + decoder)
        and applies custom functions to analyze the outputs.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        step_func
            Function to apply to the autoencoder output at each step.
            Should accept (inference_outputs, generative_outputs, losses, store)
            as arguments.
        aggregation_func
            Function to aggregate the step results from the store.
            Should accept the store list and return the final result.
        indices
            Indices of cells in adata to use. If None, all cells are used.
        batch_size
            Minibatch size for data loading into model.
            Defaults to scvi.settings.batch_size.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).

        Returns
        -------
        np.ndarray
            Aggregated results from processing all data through the autoencoder.

        Notes
        -----
        This method processes data through both the encoder and decoder components
        of the model. The step_func receives:
        - inference_outputs: Outputs from the encoder (latent variables, etc.)
        - generative_outputs: Outputs from the decoder (reconstruction parameters)
        - losses: Training losses for the batch
        - store: List for accumulating results

        When deterministic=True, the model operates without stochastic sampling,
        which is useful for reproducible analysis.

        Examples
        --------
        >>> import anndata as ad
        >>> # Define function to extract latent means
        >>> def extract_latent_means(inference_outputs, generative_outputs, losses, store):
        ...     store.append(inference_outputs["qz_m"].detach().cpu())
        >>> # Define aggregation function
        >>> def concatenate_latents(store):
        ...     return torch.cat(store, dim=0).numpy()
        >>> # Process data through autoencoder
        >>> latents = model.iterate_on_ae_output(
        ...     adata=adata, step_func=extract_latent_means, aggregation_func=concatenate_latents, deterministic=True
        ... )
        >>> print(latents.shape)  # (n_cells, n_latent)
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
        """Return the effect of each split on the reconstructed expression per sample.

        This method analyzes how different model splits contribute to the
        reconstruction of gene expression values.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        add_to_counts
            Value to add to the counts before computing the logarithm.
            Used for numerical stability in log-space calculations.
        aggregate_over_cells
            Whether to aggregate the effect over cells and return a value per dimension.
            If False, returns per-cell effects.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).
        **kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.

        Returns
        -------
        np.ndarray
            Effect of each split on reconstruction.
            Shape depends on aggregate_over_cells:
            - If True: (n_splits,) - aggregated effects per split
            - If False: (n_cells, n_splits) - per-cell effects

        Notes
        -----
        This method computes the contribution of each model split to the
        reconstruction. The calculation depends on the model's split_aggregation:

        - "logsumexp": Uses log-space softmax aggregation
        - "sum": Uses absolute value summation

        The effect is computed by analyzing how each split contributes to
        the final reconstruction parameters.

        Examples
        --------
        >>> # Get aggregated effects across all cells
        >>> effects = model.get_reconstruction_effect_of_each_split()
        >>> print(effects.shape)  # (n_splits,)
        >>> print("Effect of each split:", effects)
        >>> # Get per-cell effects
        >>> cell_effects = model.get_reconstruction_effect_of_each_split(aggregate_over_cells=False)
        >>> print(cell_effects.shape)  # (n_cells, n_splits)
        """

        def calculate_effect(inference_outputs, generative_outputs, losses, store):
            if self.module.split_aggregation == "logsumexp":
                log_mean_params = generative_outputs["original_params"]["mean"]  # n_samples x n_splits x n_genes
                log_mean_params = F.pad(
                    log_mean_params, (0, 0, 0, 1), value=np.log(add_to_counts)
                )  # n_samples x (n_splits + 1) x n_genes
                effect_share = -torch.log(1 - F.softmax(log_mean_params, dim=-2)[:, :-1, :]).sum(
                    dim=-1
                )  # n_samples x n_splits
            elif self.module.split_aggregation == "sum":
                effect_share = torch.abs(generative_outputs["original_params"]["mean"]).sum(
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

    def get_max_effect_of_splits_within_distribution(
        self,
        adata: AnnData | None = None,
        add_to_counts: float = 1.0,
        deterministic: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Return the maximum effect of each split on the reconstructed expression params for all genes.

        This method computes the maximum contribution of each split across all
        samples in the dataset, providing a global view of split importance.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        add_to_counts
            Value to add to the counts before computing the logarithm.
            Used for numerical stability in log-space calculations.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).
        **kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.

        Returns
        -------
        np.ndarray
            Max effect of each split on the reconstructed expression params for all genes


            Maximum effect of each split on the reconstructed expression params.
            Shape: (n_splits, n_genes)

        Notes
        -----
        This function is experimental. Please use interpretability pipeline or DE instead.

        The calculation depends on the model's split_aggregation:
        - "logsumexp": Uses log-space softmax aggregation
        - "sum": Uses absolute value summation

        Examples
        --------
        >>> import utils.plotting._interpretability
        >>>
        >>> # Get empirical maximum effects
        >>> max_effects = model.get_max_effect_of_splits_within_distribution(add_to_counts=0.1)
        >>> print(max_effects.shape)  # (n_splits, n_genes)
        >>>
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
                log_mean_params = generative_outputs["original_params"]["mean"]  # n_samples x n_splits x n_genes
                log_mean_params = F.pad(
                    log_mean_params, (0, 0, 0, 1), value=np.log(add_to_counts)
                )  # n_samples x (n_splits + 1) x n_genes
                effect_share = -torch.log(1 - F.softmax(log_mean_params, dim=-2)[:, :-1, :])
            elif self.module.split_aggregation == "sum":
                effect_share = torch.abs(generative_outputs["original_params"]["mean"])
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
