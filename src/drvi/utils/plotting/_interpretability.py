import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt

from drvi.utils.plotting import cmap
from drvi.utils.tools import iterate_on_top_differential_vars
from drvi.utils.tools.interpretability._latent_traverse import get_dimensions_of_traverse_data


def make_heatmap_groups(ordered_list):
    n_groups, group_names = zip(
        *[(len(list(group)), key) for (key, group) in itertools.groupby(ordered_list)], strict=False
    )
    group_positions = [0] + list(itertools.accumulate(n_groups))
    group_positions = list(zip(group_positions[:-1], [c - 1 for c in group_positions[1:]], strict=False))
    return group_positions, group_names


def differential_vars_heatmap(
    traverse_adata: AnnData,
    key: str,
    title_col: str = "title",
    score_threshold: float = 0.0,
    remove_vanished: bool = True,
    remove_unaffected: bool = False,
    figsize: tuple[int, int] | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Generate a heatmap of differential variables based on traverse data.

    Parameters
    ----------
    - traverse_adata (AnnData): Annotated data object containing traverse data.
    - key (str): Key used to access traverse effect data in `traverse_adata.varm`.
    - title_col (str): Column name in `traverse_adata.obs` to use as dimension labels.
    - score_threshold (float): Threshold value for filtering variables based on the score.
    - remove_vanished (bool): Whether to remove variables that have vanished.
    - remove_unaffected (bool): Whether to remove variables that have no effect.
    - figsize (Optional[Tuple[int, int]]): Size of the figure (width, height).
    - show (bool): Whether to show the heatmap.
    - **kwargs: Additional keyword arguments to be passed to `sc.pl.heatmap`.

    Returns
    -------
    - None if show is True, otherwise the plot.
    """
    n_latent, n_steps, n_samples, n_vars = get_dimensions_of_traverse_data(traverse_adata)

    max_effect_index_in_positive_direction_for_each_gene = np.abs(
        traverse_adata.varm[f"{key}_traverse_effect_pos"]
    ).values.argmax(axis=1)
    max_effect_in_positive_direction_for_each_gene = np.abs(
        traverse_adata.varm[f"{key}_traverse_effect_pos"].values[
            np.arange(traverse_adata.n_vars), max_effect_index_in_positive_direction_for_each_gene
        ]
    )
    max_effect_index_in_negative_direction_for_each_gene = np.abs(
        traverse_adata.varm[f"{key}_traverse_effect_neg"]
    ).values.argmax(axis=1)
    max_effect_in_negative_direction_for_each_gene = np.abs(
        traverse_adata.varm[f"{key}_traverse_effect_neg"].values[
            np.arange(traverse_adata.n_vars), max_effect_index_in_negative_direction_for_each_gene
        ]
    )
    traverse_adata.var["max_effect"] = np.maximum(
        max_effect_in_positive_direction_for_each_gene, max_effect_in_negative_direction_for_each_gene
    )
    for col in ["dim_id", "order", title_col]:
        title_mapping = dict(zip(traverse_adata.obs["dim_id"].values, traverse_adata.obs[col].values, strict=False))
        traverse_adata.var[f"max_effect_dim_{col}"] = np.where(
            traverse_adata.var["max_effect"] < score_threshold,
            float("nan") if np.isreal(traverse_adata.obs[col].values[0]) else "NONE",
            np.where(
                max_effect_in_positive_direction_for_each_gene > max_effect_in_negative_direction_for_each_gene,
                pd.Series(max_effect_index_in_positive_direction_for_each_gene).map(title_mapping),
                pd.Series(max_effect_index_in_negative_direction_for_each_gene).map(title_mapping),
            ),
        )
        traverse_adata.var[f"max_effect_dim_{col}_plus"] = np.where(
            traverse_adata.var["max_effect"] < score_threshold,
            "NONE",
            np.where(
                max_effect_in_positive_direction_for_each_gene > max_effect_in_negative_direction_for_each_gene,
                pd.Series(max_effect_index_in_positive_direction_for_each_gene).map(title_mapping).astype(str) + " +",
                pd.Series(max_effect_index_in_negative_direction_for_each_gene).map(title_mapping).astype(str) + " -",
            ),
        )

    plot_adata = AnnData(
        traverse_adata.uns[f"{key}_traverse_effect_stepwise"].reshape(n_latent * n_steps, n_vars),
        var=traverse_adata.var,
        obs=pd.DataFrame(
            {
                "dim_id": np.repeat(np.arange(n_latent), n_steps),
                "step_id": np.tile(np.arange(n_steps), n_latent),
            }
        ),
    )
    for col in ["dim_id", "order", title_col, "vanished"]:
        title_mapping = dict(zip(traverse_adata.obs["dim_id"].values, traverse_adata.obs[col].values, strict=False))
        plot_adata.obs[col] = plot_adata.obs["dim_id"].map(title_mapping)

    if remove_vanished:
        plot_adata = plot_adata[~plot_adata.obs["vanished"]].copy()
    if remove_unaffected:
        plot_adata = plot_adata[:, plot_adata.var["max_effect"] > score_threshold].copy()
    plot_adata = plot_adata[
        :, plot_adata.var.sort_values(["max_effect_dim_order", "max_effect_dim_order_plus", "max_effect"]).index
    ].copy()
    plot_adata = plot_adata[plot_adata.obs.sort_values(["order"]).index].copy()

    if figsize is None:
        figsize = (20, plot_adata.obs["dim_id"].nunique() / 4)

    vmin = min(
        -1,
        min(
            traverse_adata.varm[f"{key}_traverse_effect_pos"].values.min(),
            traverse_adata.varm[f"{key}_traverse_effect_neg"].values.min(),
        ),
    )
    vmax = max(
        +1,
        max(
            traverse_adata.varm[f"{key}_traverse_effect_pos"].values.max(),
            traverse_adata.varm[f"{key}_traverse_effect_neg"].values.max(),
        ),
    )
    var_group_positions, var_group_labels = make_heatmap_groups(plot_adata.var[f"max_effect_dim_{title_col}_plus"])
    kwargs = {
        **dict(  # noqa: C408
            vcenter=0,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap.saturated_red_blue_cmap,
            var_group_positions=var_group_positions,
            var_group_labels=var_group_labels,
            var_group_rotation=90,
        ),
        **kwargs,
    }

    return sc.pl.heatmap(
        plot_adata,
        plot_adata.var.index,
        groupby=title_col,
        layer=None,
        figsize=figsize,
        dendrogram=False,
        show=show,
        **kwargs,
    )


def _bar_plot_top_differential_vars(
    plot_info: Sequence[tuple[str, pd.Series]],
    dim_subset: Sequence[str] | None = None,
    n_top_genes: int = 10,
    ncols: int = 5,
    show: bool = True,
):
    """
    Plot the top differential variables in a bar plot.

    Parameters
    ----------
        plot_info (Sequence[Tuple[str, pd.Series]]): Information about the top differential variables.
        dim_subset (Sequence[sre]): List of dimensions to plot in the bar plot. If not specified all dimensions are plotted.
        n_top_genes (int, optional): Number of top genes to plot. Defaults to 10.
        ncols (int, optional): Number of columns in the plot grid. Defaults to 5.
        show (bool, optional): Whether to display the plot. If False, the plot will be returned as a Figure object. Defaults to True.

    Returns
    -------
        None if show is True, otherwise the figure.
    """
    if dim_subset is not None:
        plot_info = dict(plot_info)
        plot_info = [(dim_id, plot_info[dim_id]) for dim_id in dim_subset]

    n_row = int(np.ceil(len(plot_info) / ncols))
    fig, axes = plt.subplots(n_row, ncols, figsize=(3 * ncols, int(1 + 0.2 * n_top_genes) * n_row))

    for ax, info in zip(axes.flatten(), plot_info, strict=False):
        dim_title = info[0]

        top_indices = info[1].sort_values(ascending=False)[:n_top_genes]
        genes = top_indices.index
        values = top_indices.values

        # Create a horizontal bar plot
        ax.barh(genes, values, color="skyblue")
        ax.set_xlabel("Gene Score")
        ax.set_title(dim_title)
        ax.invert_yaxis()
        ax.grid(False)

    for ax in axes.flatten()[len(plot_info) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def show_top_differential_vars(
    traverse_adata: AnnData,
    key: str,
    title_col: str = "title",
    order_col: str = "order",
    dim_subset: Sequence[str] | None = None,
    gene_symbols: str | None = None,
    score_threshold: float = 0.0,
    n_top_genes: int = 10,
    ncols: int = 5,
    show: bool = True,
):
    """
    Show top differential variables in a bar plot.

    Parameters
    ----------
        traverse_adata (AnnData): Annotated data object containing the variables to be plotted.
        key (str): Key to access the traverse effect variables in `traverse_adata.varm`.
        title_col (str, optional): Column name in `traverse_adata.obs` that contains the titles for each dimension. Defaults to 'title'.
        order_col (str, optional): Column name in `traverse_adata.obs` that specifies the order of dimensions. Defaults to 'order'.  Ignored if `dim_subset` is provided.
        dim_subset (Sequence[sre]): List of dimensions to plot in the bar plot. If not specified all dimensions are plotted.
        gene_symbols (str, optional): Column name in `traverse_adata.var` that contains gene symbols. If provided, gene symbols will be used in the plot instead of gene indices. Defaults to None.
        score_threshold (float, optional): Threshold value for gene scores. Only genes with scores above this threshold will be plotted. Defaults to 0.
        n_top_genes (int, optional): Number of top genes to plot. Defaults to 10.
        ncols (int, optional): Number of columns in the plot grid. Defaults to 5.
        show (bool, optional): Whether to display the plot. If False, the plot will be returned as a Figure object. Defaults to True.

    Returns
    -------
        None if show is True, otherwise the figure.
    """
    plot_info = iterate_on_top_differential_vars(
        traverse_adata, key, title_col, order_col, gene_symbols, score_threshold
    )

    return _bar_plot_top_differential_vars(plot_info, dim_subset, n_top_genes, ncols, show)


def show_differential_vars_scatter_plot(
    traverse_adata: AnnData,
    key_x: str,
    key_y: str,
    key_combined: str,
    title_col: str = "title",
    order_col: str = "order",
    gene_symbols: str | None = None,
    score_threshold: float = 0.0,
    dim_subset: Sequence[str] | None = None,
    ncols: int = 3,
    show: bool = True,
    **kwargs,
):
    """
    Show a scatter plot of differential variables conidering multiple criteria.

    Parameters
    ----------
    - traverse_adata (AnnData): Annotated data object containing the variables to be plotted.
    - key_x (str): Key to access the first variable in `traverse_adata.varm`.
    - key_y (str): Key to access the second variable in `traverse_adata.varm`.
    - key_combined (str): Key to access the combined variable in `traverse_adata.varm`.
    - title_col (str, optional): Column name in `traverse_adata.obs` that contains the titles for each dimension. Defaults to 'title'.
    - order_col (str, optional): Column name in `traverse_adata.obs` that specifies the order of dimensions. Defaults to 'order'.  Ignored if `dim_subset` is provided.
    - gene_symbols (str, optional): Column name in `traverse_adata.var` that contains gene symbols. If provided, gene symbols will be used in the plot instead of gene indices. Defaults to None.
    - score_threshold (float, optional): Threshold value for gene scores. Only genes with scores above this threshold will be plotted. Defaults to 0.
    - dim_subset (Optional[Sequence[str]], optional): Subset of dimensions to plot. If None, all dimensions will be plotted. Defaults to None.
    - ncols (int, optional): Number of columns in the plot grid. Defaults to 3.
    - show (bool, optional): Whether to display the plot. If False, the plot will be returned as a Figure object. Defaults to True.
    - **kwargs: Additional keyword arguments to be passed to the scatter plot.

    Returns
    -------
    - None if show is True, otherwise the figure.
    """
    plot_info = {}
    for key in [key_x, key_y, key_combined]:
        plot_info[key] = iterate_on_top_differential_vars(
            traverse_adata, key, title_col, order_col, gene_symbols, score_threshold
        )

    if dim_subset is None:
        dim_ids = [dim_id for dim_id, _ in plot_info[key_combined]]
    else:
        dim_ids = [dim_id for dim_id, _ in plot_info[key_combined] if dim_id in dim_subset]
    for key in [key_x, key_y, key_combined]:
        plot_info[key] = dict(plot_info[key])

    n_plots = len(dim_ids)
    n_row = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(n_row, ncols, figsize=(5 * ncols, 4 * n_row), sharex=False, sharey=False)

    for ax, dim_id in zip(axes.flatten(), dim_ids, strict=False):
        df = (
            pd.concat(
                {
                    key_x: plot_info[key_x][dim_id],
                    key_y: plot_info[key_y][dim_id],
                    key_combined: plot_info[key_combined][dim_id],
                },
                axis=1,
            )
            .dropna()
            .sort_values(key_combined, ascending=False)
        )

        scatter = ax.scatter(df[key_x], df[key_y], c=df[key_combined], cmap="Reds", **kwargs)

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(key_combined)

        top_20 = df.nlargest(20, key_combined)
        for idx, row in top_20.iterrows():
            ax.text(row[key_x], row[key_y], str(idx), fontsize=8, ha="left", va="bottom")

        ax.set_title(dim_id)
        ax.set_xlabel(key_x)
        ax.set_ylabel(key_y)

    for ax in axes.flatten()[n_plots:]:
        fig.delaxes(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def _umap_of_relevant_genes(
    adata: AnnData,
    embed: AnnData,
    plot_info: Sequence[tuple[str, pd.Series]],
    layer: str | None = None,
    title_col: str = "title",
    gene_symbols: str | None = None,
    dim_subset: Sequence[str] | None = None,
    n_top_genes: int = 10,
    max_cells_to_plot: int | None = None,
    **kwargs,
):
    if max_cells_to_plot is not None and adata.n_obs > max_cells_to_plot:
        adata = sc.pp.subsample(adata, n_obs=max_cells_to_plot, copy=True)

    if dim_subset is not None:
        plot_info = dict(plot_info)
        plot_info = [(dim_id, plot_info[dim_id]) for dim_id in dim_subset]

    adata.obsm["X_umap_method"] = embed[adata.obs.index].obsm["X_umap"]
    for dim_title, gene_scores in plot_info:
        print(dim_title)
        relevant_genes = gene_scores.sort_values(ascending=False).index.to_list()

        if dim_title[-1] == "+":
            _cmap = cmap.saturated_sky_cmap
            real_dim_title = dim_title[:-1]
        elif dim_title[-1] == "-":
            _cmap = cmap.saturated_sky_cmap.reversed()
            real_dim_title = dim_title[:-1]
        else:
            _cmap = cmap.saturated_red_blue_cmap
            real_dim_title = dim_title

        adata.obs[dim_title] = list(embed[adata.obs.index, embed.var[title_col] == real_dim_title].X[:, 0])
        ax = sc.pl.embedding(
            adata, "X_umap_method", color=[dim_title], cmap=_cmap, vcenter=0, show=False, frameon=False, **kwargs
        )
        ax.text(0.92, 0.05, ax.get_title(), size=15, ha="left", color="black", rotation=90, transform=ax.transAxes)
        ax.set_title("")
        plt.show()

        axes = sc.pl.embedding(
            adata,
            "X_umap_method",
            layer=layer,
            color=relevant_genes[:n_top_genes],
            cmap=cmap.saturated_just_sky_cmap,
            gene_symbols=gene_symbols,
            show=False,
            frameon=False,
            **kwargs,
        )
        if n_top_genes == 1 or len(relevant_genes) == 1:
            axes = [axes]
        for ax in axes:
            ax.text(0.92, 0.05, ax.get_title(), size=15, ha="left", color="black", rotation=90, transform=ax.transAxes)
            ax.set_title("")
        plt.show()


def plot_relevant_genes_on_umap(
    adata: AnnData,
    embed: AnnData,
    traverse_adata: AnnData,
    traverse_adata_key: str,
    layer: str | None = None,
    title_col: str = "title",
    order_col: str = "order",
    gene_symbols: str | None = None,
    score_threshold: float = 0.0,
    dim_subset: Sequence[str] = None,
    n_top_genes: int = 10,
    max_cells_to_plot: int | None = None,
    **kwargs,
):
    plot_info = iterate_on_top_differential_vars(
        traverse_adata, traverse_adata_key, title_col, order_col, gene_symbols, score_threshold
    )

    return _umap_of_relevant_genes(
        adata,
        embed,
        plot_info,
        layer,
        title_col,
        gene_symbols,
        dim_subset,
        n_top_genes,
        max_cells_to_plot,
        **kwargs,
    )
