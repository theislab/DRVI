import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt

from drvi.utils.plotting import cmap
from drvi.utils.tools import iterate_on_top_differential_vars
from drvi.utils.tools.interpretability._latent_traverse import get_dimensions_of_traverse_data


def make_heatmap_groups(ordered_list: list) -> tuple[list[tuple[int, int]], list[Any]]:
    """Create group positions and labels for scanpy heatmap visualization of marker genes.

    This helper function processes an ordered list to identify groups of
    consecutive identical elements and returns their positions and labels.
    It's used to create group annotations for scanpy heatmap plots.

    Parameters
    ----------
    ordered_list
        List of elements where consecutive identical elements form groups.

    Returns
    -------
    tuple[list[tuple[int, int]], list]
        A tuple containing:
        - List of tuples with (start_index, end_index) for each group
        - List of group labels (unique values from ordered_list)

    Notes
    -----
    The function uses `itertools.groupby` to identify consecutive groups
    of identical elements. Each group is represented by its start and end
    indices (inclusive).

    Examples
    --------
    >>> # Simple example
    >>> groups, labels = make_heatmap_groups(["A", "A", "B", "B", "B", "A"])
    >>> print(f"Groups: {groups}")  # [(0, 1), (2, 4), (5, 5)]
    >>> print(f"Labels: {labels}")  # ['A', 'B', 'A']
    """
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
    """Generate a heatmap of differential variables based on traverse data.

    This function creates a comprehensive heatmap visualization showing how
    genes respond to latent dimension traversals. The heatmap displays
    stepwise effects across all latent dimensions and genes, with genes
    grouped by their maximum effect dimension.

    Parameters
    ----------
    traverse_adata
        AnnData object containing traverse data from `traverse_latent` or
        `make_traverse_adata`. Must contain differential effect data for the specified key.
    key
        Key prefix for the differential variables in `traverse_adata.varm`.
        Should correspond to a key used in `find_differential_effects` or
        `calculate_differential_vars` (e.g., "max_possible", "min_possible", "combined_score").
    title_col
        Column name in `traverse_adata.obs` to use as dimension labels.
        These titles will be used for axis labels and grouping.
    score_threshold
        Threshold value for filtering genes based on their maximum effect score.
        Only genes with maximum effects above this threshold will be included.
    remove_vanished
        Whether to remove latent dimensions that have vanished (have no effect).
        This helps focus the visualization on meaningful dimensions.
    remove_unaffected
        Whether to remove genes that have no significant effect (below score_threshold).
        When True, only genes with effects above the threshold are shown.
    figsize
        Size of the figure (width, height) in inches. If None, automatically
        calculated based on the number of dimensions.
    show
        Whether to display the plot. If False, returns the plot object.
    **kwargs
        Additional keyword arguments passed to `sc.pl.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes or None
        The heatmap plot axes if `show=False`, otherwise None.

    Raises
    ------
    KeyError
        If required data is missing from `traverse_adata`.
    ValueError
        If the specified key doesn't exist in the AnnData object.

    Notes
    -----
    The function performs the following steps:
    1. Calculates maximum effects for each gene in both positive and negative directions
    2. Identifies which dimension has the maximum effect for each gene
    3. Groups genes by their maximum effect dimension
    4. Creates a heatmap showing stepwise effects across all dimensions
    5. Applies filtering based on score threshold and vanished dimensions

    **Visualization Features:**

    - **Color scale**: Red-blue diverging colormap centered at 0
    - **Gene grouping**: Genes are grouped by their maximum effect dimension
    - **Dimension ordering**: Dimensions are ordered by their `order` column
    - **Gene ordering**: Within each group, genes are ordered by effect magnitude

    **Interpretation:**

    - **Red colors**: Positive effects (increased expression)
    - **Blue colors**: Negative effects (decreased expression)
    - **Intensity**: Magnitude of the effect
    - **Gene groups**: Genes with similar maximum effects are grouped together

    Examples
    --------
    >>> # Basic heatmap with combined scores
    >>> differential_vars_heatmap(traverse_adata, "combined_score")
    >>> # Heatmap with custom parameters
    >>> differential_vars_heatmap(
    ...     traverse_adata, "max_possible", score_threshold=1.0, remove_unaffected=True, figsize=(15, 8)
    ... )
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
    """Plot the top differential variables in a group of bar plots.

    This internal function creates horizontal bar plots showing the top genes
    for each latent dimension based on their differential effect scores.

    Parameters
    ----------
    plot_info
        Sequence of tuples containing dimension titles and corresponding gene data.
    dim_subset
        Subset of dimensions to plot. If None, all dimensions are plotted.
    n_top_genes
        Number of top genes to show in each plot.
    ncols
        Number of columns in the subplot grid.
    show
        Whether to display the plot. If False, returns the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if `show=False`, otherwise None.

    Notes
    -----
    The function creates a grid of horizontal bar plots, with each subplot
    showing the top genes for one latent dimension. Genes are sorted by
    their effect scores in descending order.

    **Plot Features:**

    - **Horizontal bars**: Gene names on y-axis, scores on x-axis
    - **Color**: Sky blue bars for all genes
    - **Grid**: No grid lines for cleaner appearance
    - **Layout**: Automatic grid layout based on number of dimensions

    Examples
    --------
    >>> # Basic bar plot
    >>> _bar_plot_top_differential_vars(plot_info)
    >>> # Custom layout
    >>> _bar_plot_top_differential_vars(plot_info, n_top_genes=15, ncols=3, show=False)
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
    """Show top differential variables in a bar plot.

    This function creates a comprehensive visualization of the top differentially
    expressed genes for each latent dimension. It generates horizontal bar plots
    showing the genes with the highest effect scores for each dimension.

    Parameters
    ----------
    traverse_adata
        AnnData object containing the differential analysis results from
        `calculate_differential_vars`. Must contain differential effect data
        for the specified key.
    key
        Key prefix for the differential variables in `traverse_adata.varm`.
        Should correspond to a key used in `find_differential_effects` or
        `calculate_differential_vars` (e.g., "max_possible", "min_possible", "combined_score").
    title_col
        Column name in `traverse_adata.obs` that contains the titles for each dimension.
        These titles will be used as subplot titles.
    order_col
        Column name in `traverse_adata.obs` that specifies the order of dimensions.
        Results will be sorted by this column. Ignored if `dim_subset` is provided.
    dim_subset
        List of dimensions to plot in the bar plot. If None, all dimensions
        with significant effects are plotted.
    gene_symbols
        Column name in `traverse_adata.var` that contains gene symbols.
        If provided, gene symbols will be used in the plot instead of gene indices.
        Useful for converting between gene IDs and readable gene names.
    score_threshold
        Threshold value for gene scores. Only genes with scores above this
        threshold will be plotted.
    n_top_genes
        Number of top genes to plot for each dimension.
    ncols
        Number of columns in the plot grid.
    show
        Whether to display the plot. If False, returns the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if `show=False`, otherwise None.

    Raises
    ------
    KeyError
        If required data is missing from `traverse_adata`.
    ValueError
        If the specified key doesn't exist in the AnnData object.

    Notes
    -----
    The function performs the following steps:
    1. Extracts top differential variables using `iterate_on_top_differential_vars`
    2. Filters dimensions based on `dim_subset` if provided
    3. Creates horizontal bar plots for each dimension
    4. Displays top `n_top_genes` genes sorted by their effect scores

    **Visualization Features:**

    - **Gene symbols**: If provided, gene symbols will be used instead of gene indices.
    - **Grid layout**: Automatic grid based on number of dimensions and `ncols`
    - **Horizontal bars**: Gene names on y-axis, scores on x-axis
    - **Color coding**: Sky blue bars for all genes
    - **Dimension titles**: Each subplot shows the dimension title
    - **Gene ordering**: Genes sorted by effect score (highest first)

    **Interpretation:**

    - **Bar length**: Represents the magnitude of the differential effect
    - **Gene position**: Higher bars indicate stronger effects
    - **Dimension separation**: Each subplot shows effects for one latent dimension
    - **Direction indicators**: Dimension titles include "+" or "-" to indicate effect direction

    Examples
    --------
    >>> # Basic visualization with combined scores
    >>> show_top_differential_vars(traverse_adata, "combined_score")
    >>> # Custom parameters with gene symbols
    >>> show_top_differential_vars(
    ...     traverse_adata, "max_possible", gene_symbols="gene_symbol", score_threshold=1.0, n_top_genes=15, ncols=3
    ... )
    >>> # Subset of dimensions
    >>> show_top_differential_vars(traverse_adata, "combined_score", dim_subset=["DR 5+", "DR 12+", "DR 14+"])
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
    """Show a scatter plot of differential variables considering multiple criteria.

    This function creates scatter plots comparing different differential effect
    (usaully "max_possible" and "min_possible") measures for each latent dimension.
    It is color-coded by the combined score. It's useful for understanding how
    different analysis methods relate to each other and identifying genes
    that show consistent effects across multiple criteria. The top 20 genes
    are labeled with their names.

    Parameters
    ----------
    traverse_adata
        AnnData object containing the differential analysis results from
        `calculate_differential_vars`. Must contain differential effect data
        for all specified keys.
    key_x
        Key for the x-axis variable in `traverse_adata.varm`.
        Typically "max_possible" or "min_possible".
    key_y
        Key for the y-axis variable in `traverse_adata.varm`.
        Typically "min_possible" or "max_possible".
    key_combined
        Key for the color-coded variable in `traverse_adata.varm`.
        Typically "combined_score" for the final combined effect.
    title_col
        Column name in `traverse_adata.obs` that contains the titles for each dimension.
        These titles will be used as subplot titles.
    order_col
        Column name in `traverse_adata.obs` that specifies the order of dimensions.
        Results will be sorted by this column. Ignored if `dim_subset` is provided.
    gene_symbols
        Column name in `traverse_adata.var` that contains gene symbols.
        If provided, gene symbols will be used for point labels instead of gene indices.
    score_threshold
        Threshold value for gene scores. Only genes with combined scores above
        this threshold will be plotted.
    dim_subset
        Subset of dimensions to plot. If None, all dimensions with significant
        effects are plotted.
    ncols
        Number of columns in the plot grid.
    show
        Whether to display the plot. If False, returns the figure object.
    **kwargs
        Additional keyword arguments passed to the scatter plot (e.g., alpha, s for point size).

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if `show=False`, otherwise None.

    Raises
    ------
    KeyError
        If required data is missing from `traverse_adata`.
    ValueError
        If any of the specified keys don't exist in the AnnData object.

    Notes
    -----
    The function performs the following steps:
    1. Extracts differential variables for all three keys (x, y, combined)
    2. Creates scatter plots for each dimension comparing the two measures
    3. Color-codes points by the combined score
    4. Labels the top 20 genes by combined score

    **Interpretation:**

    - **X-axis**: Effect measure from `key_x` (e.g., max_possible)
    - **Y-axis**: Effect measure from `key_y` (e.g., min_possible)
    - **Color**: Combined score from `key_combined`
    - **Point position**: Relationship between the two measures
    - **Labeled points**: Genes with highest combined scores
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
    """Plot UMAP embeddings for specific latent dimensions together with UMAP embeddings of relevant genes.

    This internal function creates UMAP visualizations showing how genes
    associated with specific latent dimensions are expressed across cells.
    The latent dimension values are color-coded by the latent dimension values.
    The top genes are color-coded by the gene expression.

    Parameters
    ----------
    adata
        AnnData object containing single-cell data with UMAP coordinates.
        Must have UMAP coordinates in `embed.obsm["X_umap"]`.
    embed
        AnnData object containing latent representations and dimension metadata.
        Must have columns in `.var` corresponding to `title_col`.
    plot_info
        Information about the top differential variables. Each tuple contains
        a dimension title and a pandas Series of gene scores.
    layer
        Layer name in `adata` to use for gene expression visualization.
        If None, uses `.X`.
    title_col
        Column name in `embed.var` that contains dimension titles.
    gene_symbols
        Column name in `adata.var` that contains gene symbols.
        If provided, gene symbols will be used instead of gene indices.
    dim_subset
        List of dimensions to plot. If None, all dimensions from plot_info are plotted.
    n_top_genes
        Number of top genes to visualize for each dimension.
    max_cells_to_plot
        Maximum number of cells to include in the plot. If None, all cells are plotted.
        Useful for large datasets to improve performance.
    **kwargs
        Additional keyword arguments passed to `sc.pl.embedding`.

    Returns
    -------
    None
        Displays the plots directly.

    Notes
    -----
    The function creates two types of visualizations for each dimension:
    1. **Latent dimension values**: Shows how the latent dimension varies across cells
    2. **Top gene expression**: Shows expression patterns of the top genes for that dimension

    **Visualization features:**

    - **UMAP coordinates**: Uses UMAP embedding from the embed object
    - **Cell subsetting**: Can limit number of cells for performance
    - **Gene labeling**: Shows gene names in plot titles
    - **Dimension labeling**: Shows dimension names in plot titles

    Examples
    --------
    >>> # Basic UMAP visualization
    >>> plot_info = iterate_on_top_differential_vars(traverse_adata, "combined_score")
    >>> _umap_of_relevant_genes(adata, embed, plot_info)
    >>> # With custom parameters
    >>> _umap_of_relevant_genes(
    ...     adata,
    ...     embed,
    ...     plot_info,
    ...     layer="counts",
    ...     title_col="title",
    ...     gene_symbols="gene_symbol",
    ...     n_top_genes=5,
    ...     max_cells_to_plot=5000,
    ... )
    """
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
    dim_subset: Sequence[str] | None = None,
    n_top_genes: int = 10,
    max_cells_to_plot: int | None = None,
    **kwargs,
):
    """Plot relevant genes on UMAP embedding.

    This function creates UMAP visualizations showing how genes associated
    with specific latent dimensions are expressed across cells. The latent
    dimension values are color-coded by the latent dimension values. The
    top genes are color-coded by the gene expression.

    Parameters
    ----------
    adata
        AnnData object containing single-cell data with gene expression.
        This is the original data used for training the model.
    embed
        AnnData object containing latent representations and dimension metadata.
        Must have UMAP coordinates in `embed.obsm["X_umap"]` and dimension
        information in `.var` columns.
    traverse_adata
        AnnData object containing differential analysis results from
        `calculate_differential_vars`. Must contain differential effect data
        for the specified key.
    traverse_adata_key
        Key prefix for the differential variables in `traverse_adata.varm`.
        Should correspond to a key used in `find_differential_effects` or
        `calculate_differential_vars` (e.g., "max_possible", "min_possible", "combined_score").
    layer
        Layer name in `adata` to use for gene expression visualization.
        If None, uses `.X`. Common options include "counts", "logcounts", etc..
    title_col
        Column name in `embed.var` that contains dimension titles.
        These titles will be used to match dimensions between objects.
    order_col
        Column name in `embed.var` that specifies the order of dimensions.
        Results will be sorted by this column. Ignored if `dim_subset` is provided.
    gene_symbols
        Column name in `adata.var` that contains gene symbols.
        If provided, gene symbols will be used instead of gene indices.
    score_threshold
        Threshold value for gene scores. Only genes with scores above this
        threshold will be visualized.
    dim_subset
        List of dimensions to plot. If None, all dimensions with significant
        effects are plotted.
    n_top_genes
        Number of top genes to visualize for each dimension.
    max_cells_to_plot
        Maximum number of cells to include in the plot. If None, all cells are plotted.
        Useful for large datasets to improve performance and reduce memory usage.
    **kwargs
        Additional keyword arguments passed to `sc.pl.embedding`.

    Returns
    -------
    None
        Displays the plots directly.

    Raises
    ------
    KeyError
        If required data is missing from any of the AnnData objects.
    ValueError
        If the specified key doesn't exist in traverse_adata.

    Notes
    -----
    The function performs the following steps:
    1. Extracts top differential variables using `iterate_on_top_differential_vars`
    2. For each dimension, creates two visualizations (I) UMAP of Latent dimension values across cells (II) UMAPs of Expression patterns of top genes for that dimension

    **Interpretation:**

    - **Latent dimension plots**: Show how the dimension varies across cell types
    - **Gene expression plots**: Show expression patterns of dimension-specific genes
    - **Color intensity**: Indicates magnitude of values/expression

    **Common Use Cases:**

    - **Biological validation**: Verify that latent dimensions capture meaningful biology
    - **Gene discovery**: Identify genes associated with specific processes
    - **Model interpretation**: Understand what biological processes each dimension represents
    - **Quality assessment**: Evaluate the biological relevance of the model

    Examples
    --------
    >>> # Basic UMAP visualization with combined scores
    >>> plot_relevant_genes_on_umap(adata, embed, traverse_adata, "combined_score")
    >>> # With custom parameters
    >>> plot_relevant_genes_on_umap(
    ...     adata,
    ...     embed,
    ...     traverse_adata,
    ...     "max_possible",
    ...     layer="logcounts",
    ...     gene_symbols="gene_symbol",
    ...     score_threshold=1.0,
    ...     n_top_genes=5,
    ...     max_cells_to_plot=5000,
    ... )
    >>> # Subset of dimensions
    >>> plot_relevant_genes_on_umap(
    ...     adata, embed, traverse_adata, "combined_score", dim_subset=["DR 5+", "DR 12+", "DR 14+"]
    ... )
    """
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
