from typing import Literal

import numpy as np
import pandas as pd
import scipy
from anndata import AnnData

from drvi.model import DRVI
from drvi.utils.tools.interpretability._latent_traverse import get_dimensions_of_traverse_data, traverse_latent


def find_differential_effects(
    traverse_adata: AnnData,
    method: Literal["max_possible", "min_possible"] = "max_possible",
    key_added: str = "effect",
    add_to_counts: float = 0.1,
    relax_max_by: float = 0.0,
):
    assert method in ["max_possible", "min_possible"]

    # Reorder the traverse_adata to original order
    original_traverse_adata = traverse_adata
    traverse_adata = traverse_adata[traverse_adata.obs.sort_values(["original_order"]).index].copy()
    traverse_adata = traverse_adata[:, traverse_adata.var.sort_values(["original_order"]).index].copy()

    # Get the number of latent dimensions, steps, samples, and vars
    n_latent, n_steps, n_samples, n_vars = get_dimensions_of_traverse_data(traverse_adata)

    # Get the dim_id and span values
    span_values = traverse_adata.obs["span_value"].values.reshape(n_latent, n_steps, n_samples)
    assert np.allclose(span_values, span_values.max(axis=-1, keepdims=True))
    span_values = span_values[:, :, 0]  # n_latent x n_steps
    dim_ids = traverse_adata.obs["dim_id"].values.reshape(n_latent, n_steps, n_samples)[:, 0, 0]  # n_latent

    # Get the output mean parameters in 4D format
    control_mean_param = traverse_adata.layers["control"].reshape(n_latent, n_steps, n_samples, n_vars)
    effect_mean_param = traverse_adata.layers["effect"].reshape(n_latent, n_steps, n_samples, n_vars)

    # Helper functions
    average_over_samples = lambda x: x.mean(axis=2)
    add_eps_in_count_space = lambda x: scipy.special.logsumexp(
        np.stack([x, np.log(add_to_counts) * np.ones_like(x)]), axis=0
    )
    find_relative_effect = lambda x, baseline: (
        scipy.special.logsumexp(np.stack([x, np.log(add_to_counts) * np.ones_like(x), baseline]), axis=0) - baseline
    )

    # Find DE for each sample and average over samples
    if method == "max_possible":
        diff_considering_small_values = average_over_samples(
            add_eps_in_count_space(effect_mean_param) - add_eps_in_count_space(control_mean_param)
        )  # n_latent x n_steps x n_vars
    elif method == "min_possible":
        reduce_dims = (
            1,
            2,
        )
        max_of_two = np.maximum(
            effect_mean_param.max(axis=reduce_dims, keepdims=True),
            control_mean_param.max(axis=reduce_dims, keepdims=True),
        )  # n_latent x 1 x n_samples, n_vars
        max_cumulative_possible_all = scipy.special.logsumexp(
            max_of_two, axis=0, keepdims=True
        )  # 1 x 1 x n_samples, n_vars
        max_cumulative_possible_other_dims = (
            np.log(np.exp(max_cumulative_possible_all) - np.exp(max_of_two)) - relax_max_by
        )  # n_latent x 1 x n_samples, n_vars
        max_cumulative_possible_other_dims = max_cumulative_possible_other_dims + np.zeros_like(
            effect_mean_param
        )  # n_latent x n_steps x n_samples, n_vars
        normalized_effect_mean_param = find_relative_effect(
            effect_mean_param, max_cumulative_possible_other_dims
        )  # n_latent x n_steps x n_samples, n_vars
        normalized_control_mean_param = find_relative_effect(
            control_mean_param, max_cumulative_possible_other_dims
        )  # n_latent x n_steps x n_samples, n_vars
        diff_considering_small_values = average_over_samples(
            normalized_effect_mean_param - normalized_control_mean_param
        )  # n_latent x n_steps x n_vars
    else:
        raise NotImplementedError()

    original_traverse_adata.uns[f"{key_added}_traverse_effect_stepwise"] = diff_considering_small_values

    # Find DE vars in positive and negative directions
    for effect_sign in ["pos", "neg"]:
        mask = np.where(span_values >= 0, 1, 0) if effect_sign == "pos" else np.where(span_values <= 0, 1, 0)
        max_effect = np.max(np.expand_dims(mask, axis=-1) * diff_considering_small_values, axis=1)  # n_latent x n_vars
        max_effect = pd.DataFrame(max_effect.T, index=traverse_adata.var_names, columns=dim_ids)
        original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"] = max_effect.loc[
            original_traverse_adata.var_names
        ].copy()
        original_traverse_adata.uns[f"{key_added}_traverse_effect_{effect_sign}_dim_ids"] = (
            original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"].columns.values
        )
        original_traverse_adata.varm[
            f"{key_added}_traverse_effect_{effect_sign}"
        ].columns = original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"].columns.astype(str)


def combine_differential_effects(
    traverse_adata: AnnData,
    keys: list[str],
    key_added: str,
    combine_function: callable,
):
    # Reorder the traverse_adata to original order
    original_traverse_adata = traverse_adata
    traverse_adata = traverse_adata[traverse_adata.obs.sort_values(["original_order"]).index].copy()
    traverse_adata = traverse_adata[:, traverse_adata.var.sort_values(["original_order"]).index].copy()

    # Get the number of latent dimensions, steps, samples, and vars
    n_latent, n_steps, n_samples, n_vars = get_dimensions_of_traverse_data(traverse_adata)

    # Get the dim_id and span values
    span_values = traverse_adata.obs["span_value"].values.reshape(n_latent, n_steps, n_samples)
    assert np.allclose(span_values, span_values.max(axis=-1, keepdims=True))
    span_values = span_values[:, :, 0]  # n_latent x n_steps
    dim_ids = traverse_adata.obs["dim_id"].values.reshape(n_latent, n_steps, n_samples)[:, 0, 0]  # n_latent

    # Combine effects
    combined_traverse_effect_stepwise = combine_function(
        *[traverse_adata.uns[f"{key}_traverse_effect_stepwise"] for key in keys]
    )

    original_traverse_adata.uns[f"{key_added}_traverse_effect_stepwise"] = combined_traverse_effect_stepwise

    # Find DE vars in positive and negative directions
    for effect_sign in ["pos", "neg"]:
        mask = np.where(span_values >= 0, 1, 0) if effect_sign == "pos" else np.where(span_values <= 0, 1, 0)
        max_effect = np.max(
            np.expand_dims(mask, axis=-1) * combined_traverse_effect_stepwise, axis=1
        )  # n_latent x n_vars
        max_effect = pd.DataFrame(max_effect.T, index=traverse_adata.var_names, columns=dim_ids)
        original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"] = max_effect.loc[
            original_traverse_adata.var_names
        ].copy()
        original_traverse_adata.uns[f"{key_added}_traverse_effect_{effect_sign}_dim_ids"] = (
            original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"].columns.values
        )
        original_traverse_adata.varm[
            f"{key_added}_traverse_effect_{effect_sign}"
        ].columns = original_traverse_adata.varm[f"{key_added}_traverse_effect_{effect_sign}"].columns.astype(str)


def calculate_differential_vars(traverse_adata: AnnData, **kwargs):
    print("Finding differential variables per latent dimension ...")
    find_differential_effects(traverse_adata, method="max_possible", key_added="max_possible", **kwargs)
    find_differential_effects(traverse_adata, method="min_possible", key_added="min_possible", **kwargs)

    def combine_function(min_possible, max_possible):
        # min_possible and max_possible dimensions: n_latent x n_steps x n_vars
        keep = (max_possible >= 1.0) & (
            (max_possible > max_possible.max(axis=(0, 1), keepdims=True) / 2)
            | (min_possible > min_possible.max(axis=(0, 1), keepdims=True) / 10)
        )
        score = np.where(keep, max_possible * min_possible, 0)
        return score

    # Combine scores with product
    combine_differential_effects(
        traverse_adata,
        keys=["min_possible", "max_possible"],
        key_added="combined_score",
        combine_function=combine_function,
    )


def get_split_effects(
    model: DRVI,
    embed: AnnData,
    n_steps=10 * 2,
    n_samples=100,
    traverse_kwargs=None,
    de_kwargs=None,
):
    """
    Get the split effect of a latent dimension.

    Parameters
    ----------
    model
        DRVI model object.
    embed
        latent representation of the model.
    n_steps
        Number of steps.
    n_samples
        Number of samples.
    traverse_kwargs
        Additional arguments passed to `traverse_the_latent`.
    kwargs
        Additional arguments passed to `find_differential_vars`.
    """
    if traverse_kwargs is None:
        traverse_kwargs = {}
    if de_kwargs is None:
        de_kwargs = {}

    traverse_adata = traverse_latent(model, embed, n_steps=n_steps, n_samples=n_samples, **traverse_kwargs)
    calculate_differential_vars(traverse_adata, **de_kwargs)

    return traverse_adata


def iterate_on_top_differential_vars(
    traverse_adata: AnnData,
    key: str,
    title_col: str = "title",
    order_col: str = "order",
    gene_symbols: str | None = None,
    score_threshold: float = 0.0,
):
    """
    Make an iterator of the top differential variables per latent dimension.

    Parameters
    ----------
    traverse_adata
        Anndata object with the split effects.
    key
        Key to the differential variables in traverse_adata.
    title_col
        Title columns defining title of the dimensions.
    order_col
        Order column that defines latent dimension orders.
    gene_symbols
        Column with the gene symbols in traverse_adata.var.
    score_threshold
        A threshold to filter the differential variables.
    """
    df_pos = traverse_adata.varm[f"{key}_traverse_effect_pos"].copy()
    df_neg = traverse_adata.varm[f"{key}_traverse_effect_neg"].copy()

    if gene_symbols is not None:
        gene_name_mapping = dict(zip(traverse_adata.var.index, traverse_adata.var[gene_symbols], strict=False))
    else:
        gene_name_mapping = dict(zip(traverse_adata.var.index, traverse_adata.var.index, strict=False))
    df_pos.index = df_pos.index.map(gene_name_mapping)
    df_neg.index = df_neg.index.map(gene_name_mapping)

    de_info = dict(
        **{
            (str(k) + "+"): v.sort_values(ascending=False).where(lambda x: x > score_threshold).dropna()
            for k, v in df_pos.to_dict(orient="series").items()
        },
        **{
            (str(k) + "-"): v.sort_values(ascending=False).where(lambda x: x > score_threshold).dropna()
            for k, v in df_neg.to_dict(orient="series").items()
        },
    )

    return [
        (f"{row[title_col]}{direction}", de_info[f"{row['dim_id']}{direction}"])
        for i, row in traverse_adata.obs[["dim_id", order_col, title_col]]
        .drop_duplicates()
        .sort_values(order_col)
        .iterrows()
        for direction in ["-", "+"]
        if len(de_info[f"{row['dim_id']}{direction}"]) > 0
    ]
