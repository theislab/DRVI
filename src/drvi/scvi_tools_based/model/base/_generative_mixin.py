from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import scvi
import torch
from scvi import REGISTRY_KEYS
from torch.nn import functional as F
from lightning import LightningDataModule
from tqdm import tqdm

from drvi.scvi_tools_based.module._constants import MODULE_KEYS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from anndata import AnnData

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
        lib: np.ndarray | None = None,
        batch_values: np.ndarray | None = None,
        cat_values: np.ndarray | None = None,
        cont_values: np.ndarray | None = None,
        batch_size: int = scvi.settings.batch_size,
        map_cat_values: bool = False,
    ):
        """Iterate over decoder outputs as a generator.

        This method processes latent samples through the generative model in batches
        and yields the generative outputs for each batch.

        Parameters
        ----------
        z
            Latent samples with shape (n_samples, n_latent).
        lib
            Library size array with shape (n_samples,).
            If None, defaults to 1e4 for all samples.
        batch_values
            Batch values with shape (n_samples,).
            If None, defaults to 0 for all samples.
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

        Yields
        ------
        dict[str, Any]
            Generative outputs for each batch.

        Notes
        -----
        This method operates in inference mode and processes data in batches
        to manage memory usage. The calling function should handle processing
        and aggregation of the yielded outputs.

        If map_cat_values is True, categorical values are automatically mapped
        to integers using the model's category mappings.

        Examples
        --------
        >>> import numpy as np
        >>> # Process latent samples and aggregate results
        >>> z = np.random.randn(50, 32)  # assuming 32 latent dimensions
        >>> store = []
        >>> for gen_output in model.iterate_on_decoded_latent_samples(z=z):
        ...     store.append(gen_output["params"]["mean"].detach().cpu())
        >>> result = torch.cat(store, dim=0).numpy()
        >>> print(result.shape)  # (50, n_genes)
        """
        self.module.eval()
        self.module.inspect_mode = True

        if cat_values is not None:
            if cat_values.ndim == 1:  # For a user not noticing cat_values should be 2d!
                cat_values = cat_values.reshape(-1, 1)
            if map_cat_values:
                mapped_values = np.zeros_like(cat_values)
                for i, (_label, map_keys) in enumerate(
                    self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)["mappings"].items()
                ):
                    cat_mapping = dict(zip(map_keys, range(len(map_keys)), strict=False))
                    mapped_values[:, i] = np.vectorize(cat_mapping.get)(cat_values[:, i])
                cat_values = mapped_values.astype(np.int32)

        if batch_values is not None:
            batch_values = batch_values.flatten()
            if map_cat_values:
                map_keys = self.adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY)["categorical_mapping"]
                batch_mapping = dict(zip(map_keys, range(len(map_keys)), strict=False))
                batch_values = np.vectorize(batch_mapping.get)(batch_values)
                batch_values = batch_values.astype(np.int32)
            batch_values = batch_values.reshape(-1, 1)

        try:
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
                    batch_tensor = (
                        torch.tensor(batch_values[slice])
                        if batch_values is not None
                        else torch.zeros((slice.shape[0], 1), dtype=torch.int32)
                    )

                    if self.module.__class__.__name__ == "DRVIModule":
                        inference_outputs = {
                            MODULE_KEYS.Z_KEY: z_tensor,
                        }
                        library_to_inject = lib_tensor
                    else:
                        raise NotImplementedError(f"Module {self.module.__class__.__name__} not supported.")

                    gen_input = self.module._get_generative_input(
                        tensors={
                            REGISTRY_KEYS.BATCH_KEY: batch_tensor,
                            REGISTRY_KEYS.LABELS_KEY: None,
                            REGISTRY_KEYS.CONT_COVS_KEY: cont_tensor,
                            REGISTRY_KEYS.CAT_COVS_KEY: cat_tensor,
                        },
                        inference_outputs=inference_outputs,
                        library_to_inject=library_to_inject,
                    )
                    gen_output = self.module.generative(**gen_input)
                    yield gen_output
        finally:
            self.module.inspect_mode = False

    @torch.inference_mode()
    def decode_latent_samples(
        self,
        z: np.ndarray,
        lib: np.ndarray | None = None,
        batch_values: np.ndarray | None = None,
        cat_values: np.ndarray | None = None,
        cont_values: np.ndarray | None = None,
        batch_size: int = scvi.settings.batch_size,
        map_cat_values: bool = False,
        return_in_log_space: bool = True,
    ) -> np.ndarray:
        r"""Return the distribution produced by the decoder for the given latent samples.

        This method computes :math:`p(x \mid z)`, the reconstruction distribution
        for given latent samples. It returns the mean of the reconstruction
        distribution for each sample.

        A user may use `model.get_normalized_expression` to get the normalized expression
        within distribution in count space in a more probabilistic way.

        Parameters
        ----------
        z
            Latent samples with shape (n_samples, n_latent).
        lib
            Library size array with shape (n_samples,).
            If None, defaults to 1e4 for all samples.
        batch_values
            Batch values with shape (n_samples,).
            If None, defaults to 0 for all samples.
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
        return_in_log_space
            Whether to return the means in log space.

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
        store: list[Any] = []
        for gen_output in self.iterate_on_decoded_latent_samples(
            z=z,
            lib=lib,
            batch_values=batch_values,
            cat_values=cat_values,
            cont_values=cont_values,
            batch_size=batch_size,
            map_cat_values=map_cat_values,
        ):
            if return_in_log_space:
                store.append(torch.log(gen_output[MODULE_KEYS.PX_KEY].mean).detach().cpu())
            else:
                store.append(gen_output[MODULE_KEYS.PX_KEY].mean.detach().cpu())
        return torch.cat(store, dim=0).numpy(force=True)

    @torch.inference_mode()
    def iterate_on_ae_output(
        self,
        adata: AnnData,
        datamodule: LightningDataModule | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        deterministic: bool = False,
    ):
        """Iterate over autoencoder outputs as a generator.

        This method processes data through the full autoencoder (encoder + decoder)
        and yields the outputs for each batch.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData. adata will be ignored if datamodule is provided.
        indices
            Indices of cells in adata to use. If None, all cells are used.
        batch_size
            Minibatch size for data loading into model.
            Defaults to scvi.settings.batch_size.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).

        Yields
        ------
        tuple[dict[str, Any], dict[str, Any], Any]
            Tuple of (inference_outputs, generative_outputs, losses) for each batch.

        Notes
        -----
        This method processes data through both the encoder and decoder components
        of the model. The calling function should handle processing and aggregation
        of the yielded outputs.

        When deterministic=True, the model operates without stochastic sampling,
        which is useful for reproducible analysis.

        Examples
        --------
        >>> import anndata as ad
        >>> # Process data through autoencoder and aggregate results
        >>> store = []
        >>> for inference_outputs, generative_outputs, losses in model.iterate_on_ae_output(
        ...     adata=adata, deterministic=True
        ... ):
        ...     store.append(inference_outputs["qzm"].detach().cpu())
        >>> latents = torch.cat(store, dim=0).numpy()
        >>> print(latents.shape)  # (n_cells, n_latent)
        """
        if datamodule is None:
            adata = self._validate_anndata(adata)
            data_loader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        else:
            datamodule.setup(stage="predict")
            data_loader = datamodule.predict_dataloader()
        self.module.inspect_mode = True

        try:
            if deterministic:
                self.module.fully_deterministic = True
            for tensors in tqdm(data_loader, mininterval=5.0):
                loss_kwargs = {"kl_weight": 1}
                yield self.module(tensors, loss_kwargs=loss_kwargs)
        except Exception as e:
            self.module.fully_deterministic = False
            raise e
        finally:
            self.module.fully_deterministic = False
            self.module.inspect_mode = False

    # @torch.inference_mode()  # TODO: uncomment this
    def iterate_on_effect_of_splits_within_distribution(
        self,
        adata: AnnData | None = None,
        datamodule: LightningDataModule | None = None,
        add_to_counts: float = 1.0,
        deterministic: bool = True,
        directional: bool = True,
        **kwargs: Any,
    ):
        """Iterate over the maximum effect of each split on the reconstructed expression params for all genes.

        This method computes the maximum contribution of each split across all
        samples in the dataset, providing a global view of split importance.
        For each latent dimension, effects are calculated independently for positive
        and negative values of that dimension.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData. adata will be ignored if datamodule is provided.
        add_to_counts
            Value to add to the counts before computing the logarithm.
            Used for numerical stability in log-space calculations.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).
        directional
            Whether to consider the directional effect of each split.
        **kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.

        Returns
        -------
        Generator[np.ndarray]
            Generator of maximum effect of each split on the reconstructed expression params.

        Notes
        -----
        This function is experimental. Please use interpretability pipeline or DE instead.

        The calculation depends on the model's split_aggregation:
        - "logsumexp": Uses log-space softmax aggregation
        - "sum": Uses absolute value summation
        """

        for inference_outputs, generative_outputs, losses in self.iterate_on_ae_output(
            adata=adata,
            datamodule=datamodule,
            deterministic=deterministic,
            **kwargs,
        ):
            latent = inference_outputs["qzm"]  # n_samples x n_splits (n_splits == n_latent_dims)

            if self.module.split_aggregation == "logsumexp":
                # The formula for the effect in this case uses softmax. Note: softmax(x) == softmax(x + c) for any constant c
                # So we can ignore library size normalization and -log(K) stablization in logsumexp aggregation.
                # But the constant we add (add_to_counts) depends on the total decoder effects.
                # We assume add_to_counts is added to each gene of a cell with total counts of 1e6.
                log_mean_params = generative_outputs[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY][
                    "mean"
                ]  # n_samples x n_splits x n_genes
                total_effect_per_cell = torch.logsumexp(log_mean_params, dim=[-2,-1])
                log_add_to_counts = total_effect_per_cell + np.log(add_to_counts / 1e6)
                log_mean_params = torch.concat([
                    log_mean_params, log_add_to_counts.reshape(-1, 1, 1).expand(-1, -1, log_mean_params.shape[-1])
                ], dim=-2)  # n_samples x (n_splits + 1) x n_genes
                effect_tensor = -torch.log(1 - F.softmax(log_mean_params, dim=-2)[:, :-1, :])
            elif self.module.split_aggregation == "sum":
                # TODO: consider library size for the sum aggregation case.
                effect_tensor = torch.abs(generative_outputs[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY]["mean"])
            else:
                raise NotImplementedError("Only logsumexp and sum aggregations are supported for now.")

            if directional:
                # effect_tensor: n_samples x n_splits x n_genes
                effect_tensor = effect_tensor.unsqueeze(1).expand(-1, 2, -1, -1)  # n_samples x 2 x n_splits x n_genes
                # Create masks for positive and negative values: n_samples x 2 x n_splits
                pos_neg_mask = torch.stack([(latent > 0).float(), (latent < 0).float()], dim=1).unsqueeze(-1)  # n_samples x 2 x n_latent_dims x 1

                effect_tensor = effect_tensor * pos_neg_mask

            yield effect_tensor, latent

    @torch.inference_mode()
    def get_reconstruction_effect_of_each_split(
        self,
        adata: AnnData | None = None,
        datamodule: LightningDataModule | None = None,
        add_to_counts: float = 1.0,
        aggregate_over_cells: bool = True,
        deterministic: bool = True,
        directional: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the effect of each split on the reconstructed expression per sample.

        This method analyzes how different model splits contribute to the
        reconstruction of gene expression values.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData. adata will be ignored if datamodule is provided.
        add_to_counts
            Value to add to the counts before computing the logarithm.
            Used for numerical stability in log-space calculations.
        aggregate_over_cells
            Whether to aggregate the effect over cells and return a value per dimension.
            If False, returns per-cell effects.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).
        directional
            Whether to consider the directional effect of each split.
        **kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.

        Returns
        -------
        np.ndarray
            Effect of each split on reconstruction.
            Shape depends on aggregate_over_cells and directional:
            - If True and directional=False: (n_splits,) - aggregated effects per split
            - If True and directional=True: (2, n_splits) - aggregated effects per split for positive and negative values
            - If False and directional=False: (n_cells, n_splits) - per-cell effects per split
            - If False and directional=True: (n_cells, 2, n_splits) - per-cell effects per split for positive and negative values

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

        store = None if aggregate_over_cells else []

        for effect_tensor, latent in self.iterate_on_effect_of_splits_within_distribution(
            adata=adata,
            datamodule=datamodule,
            add_to_counts=add_to_counts,
            deterministic=deterministic,
            directional=directional,
            **kwargs,
        ):
            # effect_tensor: n_samples x n_splits x n_genes
            effect_share = effect_tensor.sum(dim=-1)  # n_samples x n_splits
            effect_share = effect_share.detach().cpu()

            if aggregate_over_cells:
                effect_share = effect_share.sum(dim=0)
                store = effect_share if store is None else store + effect_share
            else:
                store.append(effect_share)

        if aggregate_over_cells:
            return store.numpy(force=True)
        else:
            return torch.cat(store, dim=0).numpy(force=True)

    # @torch.inference_mode()  # TODO: uncomment this
    def get_effect_of_splits_within_distribution(
        self,
        adata: AnnData | None = None,
        datamodule: LightningDataModule | None = None,
        add_to_counts: float = 1.0,
        deterministic: bool = True,
        directional: bool = True,
        aggregation: Literal["max", "linear_weighted_mean", "exp_weighted_mean"] = "max",
        skip_threshold: float = 1.0,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the maximum effect of each split on the reconstructed expression params for all genes.

        This method computes the maximum contribution of each split across all
        samples in the dataset, providing a global view of split importance.
        When directional=True, for each latent dimension, effects are calculated
        independently for positive and negative values of that dimension.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData.
            If None, defaults to the AnnData object used to initialize the model.
        datamodule
            LightningDataModule object with equivalent structure to initial AnnData.
            adata will be ignored if datamodule is provided.
        add_to_counts
            Value to add to the counts before computing the logarithm.
            Used for numerical stability in log-space calculations.
        deterministic
            Makes model fully deterministic (e.g., no sampling in the bottleneck).
        directional
            Whether to consider the directional effect of each split.
            If True, effects are computed separately for positive and negative
            latent values. If False, effects are computed over all samples.
        aggregation
            Aggregation method to use across batches. Currently only "max" is supported.
        **kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.

        Returns
        -------
        np.ndarray
            Maximum effect of each split on the reconstructed expression params.
            Shape depends on `directional`:
            - If `directional=True`: (2, n_splits, n_genes)
              - First dimension: 0 for positive values, 1 for negative values
              - Second dimension: split index (n_splits == n_latent_dims)
              - Third dimension: genes
            - If `directional=False`: (n_splits, n_genes)
              - First dimension: split index
              - Second dimension: genes

        Notes
        -----
        This function is experimental. Please use interpretability pipeline or DE instead.

        The calculation depends on the model's split_aggregation:
        - "logsumexp": Uses log-space softmax aggregation
        - "sum": Uses absolute value summation

        When `directional=True`, the maximum effect is computed separately:
        - Over samples where each dimension is positive (index 0 in first dimension)
        - Over samples where each dimension is negative (index 1 in first dimension)

        Examples
        --------
        >>> # Get empirical maximum effects (2, n_split, n_genes)
        >>> max_effects = model.get_max_effect_of_splits_within_distribution(add_to_counts=1.0)
        >>>
        >>> var_info = (
        ...     pd.concat(
        ...         [embed.var.assign(direction = '+'),
        ...          embed.var.assign(direction = '-')]
        ...     )
        ...     .reset_index(drop=True)
        ...     .assign(title = lambda df: df['title'] + df['direction'])
        ... )
        >>>
        >>> effect_data = (
        ...     pd.DataFrame(
        ...         np.concatenate([max_effects[0], max_effects[1]]),
        ...         columns=model.adata.var_names,
        ...         index=var_info['title'],
        ...     )
        ...     .loc[var_info.sort_values(["order", "direction"])["title"]]
        ...     .T
        ... )
        >>> plot_info = list(effect_data.to_dict(orient="series").items())
        >>> drvi.utils.plotting._interpretability._bar_plot_top_differential_vars(plot_info)
        """
        aggregated_effects = None
        normalization_factor = None
        for effect_tensor, latent in self.iterate_on_effect_of_splits_within_distribution(
            adata=adata,
            datamodule=datamodule,
            add_to_counts=add_to_counts,
            deterministic=deterministic,
            directional=directional,
            **kwargs,
        ):
            if aggregation == "max":
                effect_tensor = effect_tensor.amax(dim=0).detach().cpu().numpy(force=True)
                if aggregated_effects is None:
                    aggregated_effects = effect_tensor
                else:
                    aggregated_effects = np.maximum(aggregated_effects, effect_tensor)
            elif aggregation in ["linear_weighted_mean", "exp_weighted_mean"]:
                if directional:
                    weights = torch.stack(
                        [latent.clip(min=skip_threshold) - skip_threshold,
                         (-latent).clip(min=skip_threshold) - skip_threshold], dim=1
                         ).unsqueeze(-1)  # n_samples x 2 x n_latent_dims x 1
                else:
                    weights = latent.unsqueeze(-1) # n_samples x n_latent_dims x 1
                if aggregation == "linear_weighted_mean":
                    pass
                elif aggregation == "exp_weighted_mean":
                    weights = torch.exp(weights) - 1.0
                else:
                    raise NotImplementedError()
                effect_tensor = (effect_tensor * weights).sum(dim=0).detach().cpu().numpy(force=True)
                sum_weights = weights.sum(dim=0).detach().cpu().numpy(force=True)
                if aggregated_effects is None:
                    aggregated_effects = effect_tensor
                    normalization_factor = sum_weights
                else:
                    aggregated_effects = aggregated_effects + effect_tensor
                    normalization_factor += sum_weights
            else:
                raise NotImplementedError()

        if aggregation == "max":
            return aggregated_effects
        elif aggregation in ["linear_weighted_mean", "exp_weighted_mean"]:
            return aggregated_effects / np.maximum(normalization_factor, 1.0)
        else:
            raise NotImplementedError()

    # @torch.inference_mode()  # TODO: uncomment this
    def get_effect_of_splits_out_of_distribution(
        self,
        embed: AnnData,
        n_steps: int = 20,
        n_samples: int = 20,
        add_to_counts: float = 1.0,
        batch_size: int = scvi.settings.batch_size,
    ) -> dict[str, np.ndarray]:
        """Return the effect of each split on reconstructed expression by traversing out of distribution.

        This method efficiently computes differential effects by iterating over batch/categorical
        covariate combinations and processing each dimension separately, avoiding large sparse matrices.

        Parameters
        ----------
        embed
            AnnData object containing latent dimension statistics in `.var`.
            Must have columns: `original_dim_id`, `min`, `max`.
        n_steps
            Number of steps in the traversal. Must be even (half negative, half positive).
        n_samples
            Number of samples to generate for each step.
        add_to_counts
            Small value added to counts to avoid log(0) issues in log-space calculations.
        batch_size
            Minibatch size for data loading into model.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing:
            - "min_possible": (n_splits, n_genes) min possible LFC effects
            - "max_possible": (n_splits, n_genes) max possible LFC effects
            - "combined": (n_splits, n_genes) combined multiplicative effects
        """
        assert n_steps % 2 == 0, "n_steps must be even"
        dim_mins = np.minimum(embed.var["min"].values, 0.0)
        dim_maxs = np.maximum(embed.var["max"].values, 0.0)

        n_batch = self.summary_stats.n_batch
        n_cat_total = [n_batch] + self._get_n_cats_per_cov()

        from itertools import product
        all_cat_combinations = list(product(*[range(n) for n in n_cat_total]))

        if n_samples >= len(all_cat_combinations):
            all_cat_combinations = all_cat_combinations
            logger.info(f"Using all {len(all_cat_combinations)} combinations of batch and categorical covariates.")
        else:
            all_cat_combinations = np.random.choice(all_cat_combinations, size=n_samples, replace=False)

        n_combined = 0
        store = {'min_possible': None, 'max_possible': None, 'combined': None}
        for all_cats in all_cat_combinations:
            batch_val = all_cats[0]
            cat_vals = all_cats[1:]

            steps = np.linspace(1, 0, num=int(n_steps / 2), endpoint=False)
            span_values = np.concatenate([
                steps[:, None] * dim_maxs[None, :],  # 0 to max (increasing)
                steps[:, None] * dim_mins[None, :],  # 0 to min (decreasing)
            ], axis=0).astype(np.float32)  # n_steps x n_latent

            effect_tensors = []
            for gen_output in self.iterate_on_decoded_latent_samples(
                z=span_values,
                lib=np.ones(n_steps) * 1e4,
                batch_values=np.full(n_steps, batch_val),
                cat_values=np.array([cat_vals] * n_steps) if cat_vals else None,
                cont_values=None,
                batch_size=batch_size,
                map_cat_values=False,
            ):
                effect_tensors.append(gen_output[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY]["mean"])  # n_batch_size x n_splits x n_genes
            effect_tensors = torch.cat(effect_tensors, dim=0)  # n_steps x n_splits x n_genes
            # distinguish positive and negative directions -> 2 x (n_steps/2) x n_splits x n_genes
            effect_tensors = effect_tensors.reshape(2, int(n_steps/2), effect_tensors.shape[1], effect_tensors.shape[2])

            directional_min_per_split = effect_tensors.amin(dim=1) # 2 x n_splits x n_genes
            directional_max_per_split = effect_tensors.amax(dim=1) # 2 x n_splits x n_genes
            min_per_split = directional_min_per_split.amin(dim=0) # n_splits x n_genes
            max_per_split = directional_max_per_split.amax(dim=0) # n_splits x n_genes
            # We assume to add 1 count out of 1e6 counts to max_possible effect for each gene
            log_add_to_counts = torch.logsumexp(max_per_split, dim=[0, 1]) + np.log(add_to_counts / 1e6)
            # Next two vars are log(sum_j(exp(z_j_min))) and log(sum_j(exp(z_j_max)))
            lse_min = torch.logsumexp(
                torch.concat([min_per_split, log_add_to_counts.reshape(1, 1).expand(1, min_per_split.shape[1])], dim=0), 
                dim=0, keepdim=True)  # 1 x n_genes
            lse_max = torch.logsumexp(
                torch.concat([max_per_split, log_add_to_counts.reshape(1, 1).expand(1, max_per_split.shape[1])], dim=0),
                dim=0, keepdim=True)  # 1 x n_genes
            # Now we calculate effects
            # max_possible LFC = log(sum_j(exp(z_j_min)) - exp(z_i_min) + exp(z_i_max)) - log(sum_j(exp(z_j_min)))
            max_possible = (
                torch.log(torch.exp(lse_min) - torch.exp(min_per_split) + torch.exp(directional_max_per_split))
                - 
                lse_min  # == torch.log(torch.exp(lse_min) - torch.exp(min_per_split) + torch.exp(min_per_split))
            )
            # min_possible LFC = log(sum_j(exp(z_j_max))) - log(sum_j(exp(z_j_max)) - exp(z_i_max) + exp(z_i_min))
            min_possible = (
                lse_max  # == torch.log(torch.exp(lse_max) - torch.exp(max_per_split) + torch.exp(max_per_split))
                - 
                torch.log(torch.exp(lse_max) - torch.exp(max_per_split) + torch.exp(directional_min_per_split))
            )
            # Add multiplicative combined effect
            combined = max_possible * min_possible

            # Aggregation: accumulate across batch/cat combinations
            if n_combined == 0:
                store['min_possible'] = min_possible
                store['max_possible'] = max_possible
                store['combined'] = combined
            else:
                store['min_possible'] = store['min_possible'] + min_possible
                store['max_possible'] = store['max_possible'] + max_possible
                store['combined'] = store['combined'] + combined
            n_combined += 1
        
        # Average across combinations
        
        store['min_possible'] = store['min_possible'] / n_combined
        store['max_possible'] = store['max_possible'] / n_combined
        store['combined'] = store['combined'] / n_combined
        # Convert to numpy
        store = {k: v.detach().cpu().numpy() for k, v in store.items()}

        return store
