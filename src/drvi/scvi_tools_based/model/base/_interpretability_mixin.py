from __future__ import annotations

import inspect
import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scvi
import torch
from lightning import LightningDataModule
from matplotlib import pyplot as plt
from scipy import sparse
from torch.nn import functional as F

from drvi.scvi_tools_based.module._constants import MODULE_KEYS
from drvi.scvi_tools_based.util._sparse_arrays import _flatten_to_1d, _sparse_std

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class InterpretabilityMixin:
    """Mixin class for interpretability in DRVI model."""

    @torch.inference_mode()
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
        for inference_outputs, generative_outputs, _losses in self.iterate_on_ae_output(
            adata=adata,
            datamodule=datamodule,
            deterministic=deterministic,
            **kwargs,
        ):
            latent = inference_outputs[MODULE_KEYS.QZM_KEY]  # n_samples x n_splits (n_splits == n_latent_dims)

            if self.module.split_aggregation == "logsumexp":
                # The formula for the effect in this case uses softmax. Note: softmax(x) == softmax(x + c) for any constant c
                # So we can ignore library size normalization and -log(K) stabilization in logsumexp aggregation.
                # But the constant we add (add_to_counts) depends on the total decoder effects.
                # We assume add_to_counts is added to each gene of a cell with total counts of 1e6.
                log_mean_params = generative_outputs[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY][
                    "mean"
                ]  # n_samples x n_splits x n_genes
                total_effect_per_cell = torch.logsumexp(log_mean_params, dim=[-2, -1])
                log_add_to_counts = total_effect_per_cell + np.log(add_to_counts / 1e6)
                log_mean_params = torch.cat(
                    [log_mean_params, log_add_to_counts.reshape(-1, 1, 1).expand(-1, -1, log_mean_params.shape[-1])],
                    dim=-2,
                )  # n_samples x (n_splits + 1) x n_genes
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
                pos_neg_mask = torch.stack([(latent > 0).float(), (latent < 0).float()], dim=1).unsqueeze(
                    -1
                )  # n_samples x 2 x n_latent_dims x 1

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

        for effect_tensor, _latent in self.iterate_on_effect_of_splits_within_distribution(
            adata=adata,
            datamodule=datamodule,
            add_to_counts=add_to_counts,
            deterministic=deterministic,
            directional=directional,
            **kwargs,
        ):
            # effect_tensor: n_samples x n_splits x n_genes or n_samples x 2 x n_splits x n_genes if directional
            effect_share = effect_tensor.sum(dim=-1)  # n_samples x n_splits or n_samples x 2 x n_splits
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

    @torch.inference_mode()
    def set_latent_dimension_stats(
        self,
        embed: AnnData,
        vanished_threshold: float = 0.5,
    ) -> AnnData | None:
        """Set the latent dimension statistics of a DRVI embedding into var of an AnnData.

        Computes and stores various statistics for each latent dimension in the
        embedding AnnData object: reconstruction effects, ordering, and basic
        statistical measures (mean, std, min, max) for each dimension.

        Parameters
        ----------
        embed
            AnnData object containing the latent representation (embedding) of the model.
            The latent dimensions should be in the `.X` attribute.
        vanished_threshold
            Threshold for determining if a latent dimension has "vanished" (become inactive).
            Dimensions with max absolute values below this threshold are marked as vanished.

        Returns
        -------
        None

        Notes
        -----
        The following columns are added to `embed.var`:

        - `original_dim_id`: Original dimension indices
        - `reconstruction_effect`: Reconstruction effect scores from the DRVI model
        - `order`: Ranking of dimensions by reconstruction effect (descending)
        - `max_value`: Maximum absolute value across all cells for each dimension
        - `mean`: Mean value across all cells for each dimension
        - `min`: Minimum value across all cells for each dimension
        - `max`: Maximum value across all cells for each dimension
        - `std`: Standard deviation; `std_abs`: std of absolute values
        - `title`: Dimension titles in format "DR {order+1}"
        - `vanished`: Boolean indicating if dimension is "vanished" (max_value < threshold)
        - `vanished_positive_direction`: Dimension is "vanished" in the + direction if max < threshold.
        - `vanished_negative_direction`: Dimension is "vanished" in the - direction if min > -threshold.

        Examples
        --------
        >>> latent_adata = model.get_latent_representation(adata, return_anndata=True)
        >>> model.set_latent_dimension_stats(latent_adata)
        >>> print(latent_adata.var[["order", "reconstruction_effect", "vanished"]].head())
        """
        if "original_dim_id" not in embed.var:
            embed.var["original_dim_id"] = np.arange(embed.var.shape[0])

        embed.var["reconstruction_effect"] = 0.0
        embed.var.loc[embed.var.sort_values("original_dim_id").index, "reconstruction_effect"] = (
            self.get_reconstruction_effect_of_each_split()
        )
        embed.var["order"] = (-embed.var["reconstruction_effect"]).argsort().argsort()

        embed.var["max_value"] = _flatten_to_1d(np.abs(embed.X).max(axis=0))
        embed.var["mean"] = _flatten_to_1d(embed.X.mean(axis=0))
        embed.var["min"] = _flatten_to_1d(embed.X.min(axis=0))
        embed.var["max"] = _flatten_to_1d(embed.X.max(axis=0))
        if sparse.issparse(embed.X):
            embed.var["std"] = _sparse_std(embed.X, axis=0)
            embed.var["std_abs"] = _sparse_std(np.abs(embed.X), axis=0)
        else:
            embed.var["std"] = embed.X.std(axis=0)
            embed.var["std_abs"] = np.abs(embed.X).std(axis=0)

        embed.var["title"] = "DR " + (1 + embed.var["order"]).astype(str)
        embed.var["vanished"] = embed.var["max_value"] < vanished_threshold
        embed.var["vanished_positive_direction"] = embed.var["max"] < vanished_threshold
        embed.var["vanished_negative_direction"] = embed.var["min"] > -vanished_threshold

    @torch.inference_mode()
    def get_effect_of_splits_within_distribution(
        self,
        adata: AnnData | None = None,
        datamodule: LightningDataModule | None = None,
        add_to_counts: float = 1.0,
        deterministic: bool = True,
        directional: bool = True,
        aggregations: Sequence[Literal["max", "linear_weighted_mean", "exp_weighted_mean"]] | str = "ALL",
        skip_threshold: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
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
        aggregations
            Aggregation methods to use across batches.
            If "ALL", all methods are used.
            If a string, only the method is used.
            If a sequence, only the methods in the sequence are used.
        skip_threshold
            Minimum threshold for latent values when computing weighted means.
            Values below this threshold are clipped before computing weights.
        **kwargs
            Additional keyword arguments for the `iterate_on_ae_output` method.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing:
            - "{aggregation_key}": (n_splits, n_genes) or (2, n_splits, n_genes) score for each split for each gene

        Notes
        -----
        The calculation depends on the model's split_aggregation:
        - "logsumexp": Uses log-space softmax aggregation
        - "sum": Uses absolute value summation

        When `directional=True`, the score is computed separately:
        - Over samples where each dimension is positive (index 0 in first dimension)
        - Over samples where each dimension is negative (index 1 in first dimension)

        Examples
        --------
        >>> # Get empirical scores (2, n_split, n_genes)
        >>> scores = model.get_effect_of_splits_within_distribution(add_to_counts=1.0)
        >>>
        >>> var_info = (
        ...     pd.concat([embed.var.assign(direction="+"), embed.var.assign(direction="-")])
        ...     .reset_index(drop=True)
        ...     .assign(title=lambda df: df["title"] + df["direction"])
        ... )
        >>>
        >>> effect_data = (
        ...     pd.DataFrame(
        ...         np.concatenate([scores["max_possible"][0], scores["max_possible"][1]]),
        ...         columns=model.adata.var_names,
        ...         index=var_info["title"],
        ...     )
        ...     .loc[var_info.sort_values(["order", "direction"])["title"]]
        ...     .T
        ... )
        >>> plot_info = list(effect_data.to_dict(orient="series").items())
        >>> drvi.utils.plotting._interpretability._bar_plot_top_differential_vars(plot_info)
        """
        if aggregations == "ALL":
            aggregations = ["max", "linear_weighted_mean", "exp_weighted_mean"]
        elif isinstance(aggregations, str):
            aggregations = [aggregations]
        store = {}
        for i, (effect_tensor, latent) in enumerate(
            self.iterate_on_effect_of_splits_within_distribution(
                adata=adata,
                datamodule=datamodule,
                add_to_counts=add_to_counts,
                deterministic=deterministic,
                directional=directional,
                **kwargs,
            )
        ):
            for aggregation in aggregations:
                if aggregation == "max":
                    effect_tensor_agg = effect_tensor.amax(dim=0).detach().cpu().numpy(force=True)
                    if i == 0:
                        store[aggregation] = {"result": effect_tensor_agg}
                    else:
                        store[aggregation]["result"] = np.maximum(store[aggregation]["result"], effect_tensor_agg)
                elif aggregation in ["linear_weighted_mean", "exp_weighted_mean"]:
                    # Calculate weights
                    if directional:
                        weights = torch.stack(
                            [
                                latent.clamp(min=skip_threshold) - skip_threshold,
                                (-latent).clamp(min=skip_threshold) - skip_threshold,
                            ],
                            dim=1,
                        ).unsqueeze(-1)  # n_samples x 2 x n_latent_dims x 1
                    else:
                        weights = latent.unsqueeze(-1)  # n_samples x n_latent_dims x 1
                    if aggregation == "exp_weighted_mean":
                        weights = torch.exp(weights) - 1.0
                    # Calculate weighted mean
                    effect_tensor_agg = (effect_tensor * weights).sum(dim=0).detach().cpu().numpy(force=True)
                    sum_weights = weights.sum(dim=0).detach().cpu().numpy(force=True)
                    # Store results
                    if i == 0:
                        store[aggregation] = {"result": effect_tensor_agg, "sum_weights": sum_weights}
                    else:
                        store[aggregation]["result"] = store[aggregation]["result"] + effect_tensor_agg
                        store[aggregation]["sum_weights"] = store[aggregation]["sum_weights"] + sum_weights
                else:
                    raise NotImplementedError()

        results = {}
        for aggregation in aggregations:
            if aggregation == "max":
                results[aggregation] = store[aggregation]["result"]
            elif aggregation in ["linear_weighted_mean", "exp_weighted_mean"]:
                results[aggregation] = store[aggregation]["result"] / np.maximum(store[aggregation]["sum_weights"], 1.0)
            else:
                raise NotImplementedError()
        return results

    @torch.inference_mode()
    def get_effect_of_splits_out_of_distribution(
        self,
        embed: AnnData,
        n_steps: int = 20,
        n_samples: int = 100,
        add_to_counts: float = 1.0,
        directional: bool = True,
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
        directional
            Whether to consider the directional effect of each split.
        batch_size
            Minibatch size for data loading into model.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing:
            - "min_possible": (n_splits, n_genes) or (2, n_splits, n_genes) min possible LFC effects
            - "max_possible": (n_splits, n_genes) or (2, n_splits, n_genes) max possible LFC effects
            - "combined": (n_splits, n_genes) or (2, n_splits, n_genes) combined multiplicative effects
        """
        assert n_steps % 2 == 0, "n_steps must be even"
        dim_mins = np.minimum(embed.var["min"].values, 0.0)
        dim_maxs = np.maximum(embed.var["max"].values, 0.0)

        n_batch = self.summary_stats.n_batch
        n_cat_total = [n_batch] + self._get_n_cats_per_cov()

        all_cat_combinations = np.asarray(list(itertools.product(*[range(n) for n in n_cat_total])))

        if n_samples >= len(all_cat_combinations):
            logger.info(f"Using all {len(all_cat_combinations)} combinations of batch and categorical covariates.")
        all_cat_combinations = np.random.permutation(all_cat_combinations)[:n_samples]

        n_combined = 0
        store = {"min_possible": None, "max_possible": None, "combined": None}
        for all_cats in all_cat_combinations:
            batch_val = all_cats[0].tolist()
            cat_vals = all_cats[1:].tolist()

            steps = np.linspace(1, 0, num=int(n_steps / 2), endpoint=False)
            span_values = np.concatenate(
                [
                    steps[:, None] * dim_maxs[None, :],  # 0 to max (increasing)
                    steps[:, None] * dim_mins[None, :],  # 0 to min (decreasing)
                ],
                axis=0,
            ).astype(np.float32)  # n_steps x n_latent

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
                effect_tensors.append(
                    gen_output[MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY]["mean"]
                )  # n_batch_size x n_splits x n_genes
            effect_tensors = torch.cat(effect_tensors, dim=0)  # n_steps x n_splits x n_genes
            # distinguish positive and negative directions -> 2 x (n_steps/2) x n_splits x n_genes
            effect_tensors = effect_tensors.reshape(
                2, int(n_steps / 2), effect_tensors.shape[1], effect_tensors.shape[2]
            )

            min_per_split = effect_tensors.amin(dim=[0, 1])  # n_splits x n_genes
            max_per_split = effect_tensors.amax(dim=[0, 1])  # n_splits x n_genes
            if directional:
                directional_max_per_split = effect_tensors.amax(dim=1)  # 2 x n_splits x n_genes
            else:
                directional_max_per_split = max_per_split  # n_splits x n_genes
            # We assume to add add_to_counts counts out of 1e6 counts to max_possible effect for each gene
            log_add_to_counts = torch.logsumexp(max_per_split, dim=[0, 1]) + np.log(add_to_counts / 1e6)
            # Next two vars are log(sum_j(exp(z_j_min))) and log(sum_j(exp(z_j_max)))
            lse_min = torch.logsumexp(
                torch.cat([min_per_split, log_add_to_counts.reshape(1, 1).expand(1, min_per_split.shape[1])], dim=0),
                dim=0,
                keepdim=True,
            )  # 1 x n_genes
            lse_max = torch.logsumexp(
                torch.cat([max_per_split, log_add_to_counts.reshape(1, 1).expand(1, max_per_split.shape[1])], dim=0),
                dim=0,
                keepdim=True,
            )  # 1 x n_genes
            # Now we calculate effects
            # max_possible LFC = log(sum_j(exp(z_j_min)) - exp(z_i_min) + exp(z_i_max)) - log(sum_j(exp(z_j_min)))
            max_possible = (
                torch.log(torch.exp(lse_min) - torch.exp(min_per_split) + torch.exp(directional_max_per_split))
                - lse_min  # == torch.log(torch.exp(lse_min) - torch.exp(min_per_split) + torch.exp(min_per_split))
            )
            # min_possible LFC = log(sum_j(exp(z_j_max))) - log(sum_j(exp(z_j_max)) - exp(z_i_max) + exp(z_i_min))
            min_possible = torch.log(
                torch.exp(lse_max) - torch.exp(max_per_split) + torch.exp(directional_max_per_split)
            ) - torch.log(torch.exp(lse_max) - torch.exp(max_per_split) + torch.exp(min_per_split))
            # Add multiplicative combined effect
            combined = max_possible * min_possible

            # Aggregation: accumulate across batch/cat combinations
            if n_combined == 0:
                store["min_possible"] = min_possible
                store["max_possible"] = max_possible
                store["combined"] = combined
            else:
                store["min_possible"] = store["min_possible"] + min_possible
                store["max_possible"] = store["max_possible"] + max_possible
                store["combined"] = store["combined"] + combined
            n_combined += 1

        # Average across combinations

        store["min_possible"] = store["min_possible"] / n_combined
        store["max_possible"] = store["max_possible"] / n_combined
        store["combined"] = store["combined"] / n_combined
        # Convert to numpy
        store = {k: v.detach().cpu().numpy() for k, v in store.items()}

        return store

    def calculate_interpretability_scores(
        self,
        embed: AnnData,
        methods: Sequence[str] | str = "OOD",
        directional: bool = True,
        add_to_counts: float = 1.0,
        inplace: bool = True,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | None:
        """Calculate interpretability scores for each split.

        Parameters
        ----------
        embed : AnnData
            AnnData object containing latent dimension statistics in `.var`.
            Must have columns: `original_dim_id`, `min`, `max`.
        methods : str or Sequence[str], optional
            Options are:
            - "ALL": all methods are used
            - "IND": in-distribution interpretability methods are used
            - "OOD": out-of-distribution interpretability methods are used
            - A sequence of specific method names
        directional : bool, optional
            Whether to consider the directional effect of each split.
        add_to_counts : float, optional
            Value to add to the counts before computing the logarithm.
            Used for numerical stability in log-space calculations.
        inplace : bool, optional
            Whether to add the results to the embed.varm in place.
            If False, returns a dictionary instead.
        **kwargs: Any
            Additional keyword arguments for the `get_effect_of_splits_within_distribution` or `get_effect_of_splits_out_of_distribution` methods.

        Returns
        -------
        dict[str, np.ndarray] | None
            If `inplace=False`, returns a dictionary containing interpretability scores for each method.
            Keys are formatted as "{method}_{aggregation}_{direction}" where direction
            is "positive" or "negative" if directional=True, otherwise omitted.
            If `inplace=True`, returns None and stores results in `embed.varm`.
        """
        if methods == "ALL":
            methods = ["IND", "OOD"]
        elif isinstance(methods, str):
            methods = [methods]
        calculate_ind = any(method.startswith("IND") for method in methods)
        calculate_ood = any(method.startswith("OOD") for method in methods)

        all_results = {}
        if calculate_ind:
            sig = inspect.signature(self.get_effect_of_splits_within_distribution)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            result_dict = self.get_effect_of_splits_within_distribution(
                directional=directional, add_to_counts=add_to_counts, aggregations="ALL", **valid_kwargs
            )
            for key, value in result_dict.items():
                if key not in methods and "IND" not in methods:
                    continue
                if directional:
                    all_results[f"IND_{key}_positive"] = value[0]
                    all_results[f"IND_{key}_negative"] = value[1]
                else:
                    all_results[f"IND_{key}"] = value
        if calculate_ood:
            sig = inspect.signature(self.get_effect_of_splits_out_of_distribution)
            valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
            result_dict = self.get_effect_of_splits_out_of_distribution(
                embed=embed, directional=directional, add_to_counts=add_to_counts, **valid_kwargs
            )
            for key, value in result_dict.items():
                if key not in methods and "OOD" not in methods:
                    continue
                if directional:
                    all_results[f"OOD_{key}_positive"] = value[0]
                    all_results[f"OOD_{key}_negative"] = value[1]
                else:
                    all_results[f"OOD_{key}"] = value

        if inplace:
            for key, value in all_results.items():
                if key in embed.varm:
                    logger.warning(f"Key {key} already exists in embed.varm, overwriting.")
                embed.varm[key] = value
        else:
            return all_results

    def get_interpretability_scores(
        self,
        embed: AnnData,
        adata: AnnData,
        key: str = "OOD_combined",
        directional: bool = True,
        gene_symbols: str | None = None,
        order_col: str = "order",
        title_col: str = "title",
        hide_vanished: bool = True,
    ) -> pd.DataFrame:
        """Extract interpretability scores as a DataFrame.

        Parameters
        ----------
        embed
            AnnData object containing interpretability scores in `.varm`.
            For directional=True, expects `{key}_positive` and `{key}_negative` keys.
        adata
            AnnData object for gene information.
        key
            Base key name for scores in `embed.varm`. Default: "OOD_combined".
        directional
            Whether to include directional effects. If True, creates columns like
            "DR 1+", "DR 1-". If False, creates columns like "DR 1".
        gene_symbols
            Column name in `adata.var` for gene symbols. If None, uses `adata.var_names`.
        order_col
            Column name in `embed.var` for dimension ordering.
        title_col
            Column name in `embed.var` for dimension titles.
        hide_vanished: bool, optional
            Whether to hide vanished dimensions from the plot.

        Returns
        -------
        pd.DataFrame
            DataFrame with genes as rows and dimensions as columns.
        """
        if directional:
            effect_data = np.concatenate([embed.varm[key + "_positive"], embed.varm[key + "_negative"]])
            var_info = (
                pd.concat([embed.var.assign(direction="+"), embed.var.assign(direction="-")])
                .assign(title=lambda df: df[title_col] + df["direction"])
                .assign(keep=True)
                .reset_index(drop=True)
            )
            var_info["keep"] = ~np.where(
                var_info["direction"] == "+", 
                var_info["vanished_positive_direction"], 
                var_info["vanished_negative_direction"]
            ) if hide_vanished else True
        else:
            effect_data = embed.varm[key]
            var_info = embed.var.assign(title=lambda df: df[title_col]).assign(direction="")
            var_info["keep"] = ~var_info["vanished"] if hide_vanished else True

        gene_names = adata.var_names if gene_symbols is None else adata.var[gene_symbols]

        return (
            pd.DataFrame(
                effect_data,
                columns=gene_names,
                index=var_info["title"],
            )
            .loc[var_info.query("keep == True").sort_values([order_col, "direction"])["title"]]
            .T
        )

    def plot_interpretability_scores(
        self,
        embed: AnnData,
        adata: AnnData,
        ncols: int = 5,
        n_top_genes: int = 10,
        score_threshold: float = 0.1,
        dim_subset: Sequence[str] | None = None,
        show: bool = True,
        **kwargs,
    ):
        """Plot interpretability scores as horizontal bar plots.

        Parameters
        ----------
        embed
            AnnData object containing interpretability scores.
        adata
            AnnData object for gene information.
        ncols
            Number of columns in the subplot grid.
        n_top_genes
            Number of top genes to display per dimension.
        score_threshold
            Minimum score threshold for dimensions to be plotted.
        dim_subset
            Optional list of dimension titles to plot. If None, all dimensions
            meeting the threshold are plotted.
        show
            Whether to display the plot. If False, returns the figure.
        **kwargs
            Additional arguments passed to `get_interpretability_scores`.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object if `show=False`, otherwise None.
        """
        plot_df = self.get_interpretability_scores(embed=embed, adata=adata, **kwargs)
        plot_info = [
            (k, v)
            for k, v in plot_df.to_dict(orient="series").items()
            if (v.max() >= score_threshold) and (dim_subset is None or k in dim_subset)
        ]

        n_row = int(np.ceil(len(plot_info) / ncols))
        fig, axes = plt.subplots(n_row, ncols, figsize=(3 * ncols, int(1 + 0.2 * n_top_genes) * n_row))

        for ax, info in zip(axes.flatten(), plot_info, strict=False):
            top_indices = info[1].sort_values(ascending=False)[:n_top_genes]
            if len(top_indices) > 0:
                ax.barh(top_indices.index, top_indices.values, color="skyblue")
                ax.set_xlabel("Gene Score")
                ax.set_title(info[0])
                ax.invert_yaxis()
            ax.grid(False)

        for ax in axes.flatten()[len(plot_info) :]:
            fig.delaxes(ax)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            return fig
