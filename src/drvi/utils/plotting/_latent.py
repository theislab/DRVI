from collections.abc import Sequence
from typing import Literal

import anndata as ad
import numpy as np
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt

from drvi.utils.plotting import cmap


def make_balanced_subsample(adata: AnnData, col: str, min_count: int = 10) -> AnnData:
    """Create a balanced subsample of AnnData based on a categorical column.

    This function creates a balanced subsample by sampling an equal number of cells
    from each category in the specified column, ensuring balanced representation.

    Parameters
    ----------
    adata
        Annotated data object to subsample.
    col
        Column name in `adata.obs` containing categorical labels for balancing.
    min_count
        Minimum number of samples per category. If a category has fewer samples
        than this, sampling will be done with replacement.

    Returns
    -------
    AnnData
        Balanced subsample of the input AnnData object.

    Notes
    -----
    The function uses a fixed random state (0) for reproducible results.
    If a category has fewer samples than `min_count`, sampling is done with replacement.
    """
    n_sample_per_cond = adata.obs[col].value_counts().min()
    balanced_sample_index = (
        adata.obs.groupby(col)
        .sample(n=max(min_count, n_sample_per_cond), random_state=0, replace=n_sample_per_cond < min_count)
        .index
    )
    adata = adata[balanced_sample_index].copy()
    return adata


def plot_latent_dimension_stats(
    embed: AnnData,
    figsize: tuple[int, int] = (5, 3),
    log_scale: bool | Literal["try"] = "try",
    ncols: int = 5,
    columns: Sequence[str] = ("reconstruction_effect", "max_value", "mean", "std"),
    titles: dict[str, str] | None = None,
    remove_vanished: bool = False,
    show: bool = True,
):
    """Plot the statistics of latent dimensions.

    This function creates line plots showing various statistics of latent dimensions
    across their ranking order. It can optionally distinguish between vanished and
    non-vanished dimensions.

    Parameters
    ----------
    embed
        Annotated data object containing the latent dimensions and their statistics
        in the `.var` attribute.
    figsize
        The size of each subplot (width, height) in inches.
    log_scale
        Whether to use a log scale for the y-axis. If "try", log scale is used
        only if the minimum value is greater than 0.
    ncols
        The maximum number of columns in the subplot grid.
    columns
        The columns from `embed.var` to plot. These should be numeric columns
        containing dimension statistics.
    titles
        Custom titles for each column in the plot. If None, default titles are used.
    remove_vanished
        Whether to exclude vanished dimensions from the plot.
    show
        Whether to display the plot. If False, returns the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
        The matplotlib figure object if `show=False`, otherwise None.

    Notes
    -----
    The function expects the following columns in `embed.var`:
    - `order`: Ranking of dimensions
    - `vanished`: Boolean indicating vanished dimensions
    - The columns specified in the `columns` parameter

    If `remove_vanished=False`, a legend is added to distinguish between
    vanished (black dots) and non-vanished (blue dots) dimensions.

    Examples
    --------
    >>> # Default plot
    >>> plot_latent_dimension_stats(embed)
    >>>
    >>> # Plot basic statistics
    >>> plot_latent_dimension_stats(embed, columns=["reconstruction_effect", "max_value"])
    >>> # Plot with custom titles and log scale
    >>> titles = {"reconstruction_effect": "Reconstruction Impact", "max_value": "Max Activation"}
    >>> plot_latent_dimension_stats(embed, titles=titles, log_scale=True)
    """
    if titles is None:
        titles = {
            "reconstruction_effect": "Reconstruction effect",
            "max_value": "Max value",
            "mean": "Mean",
            "std": "Standard Deviation",
        }
    nrows = int(np.ceil(len(columns) / ncols))
    if nrows == 1:
        ncols = len(columns)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), sharey=False, sharex=False, squeeze=False
    )

    # Iterate through columns and plot the data
    for ax, col in zip(axes.flatten(), columns, strict=False):
        df = embed.var
        if remove_vanished:
            df = df.query("vanished == False")
        df = df.sort_values("order")
        ranks = df["order"]
        x = df[col]

        ax.plot(ranks, x, linestyle="-", color="grey", label="Line")  # Solid line plot
        for vanished_status_to_plot in [True, False]:
            indices = df["vanished"] == vanished_status_to_plot
            ax.plot(
                ranks[indices],
                x[indices],
                "o",
                markersize=3,
                color="black" if vanished_status_to_plot else "blue",
                label="Data Points",
            )

        # Adding labels and title
        ax.set_xlabel("Rank based on Explanation Share")
        ax.set_ylabel(titles[col] if col in titles else col)
        if isinstance(log_scale, str):
            if log_scale == "try":
                if x.min() > 0:
                    ax.set_yscale("log")
        else:
            if log_scale:
                ax.set_yscale("log")

        # Removing the legend
        ax.legend().remove()

        # Adding grid
        ax.grid(axis="x")

    if not remove_vanished:
        # Create custom legend entries
        handles = []
        for vanished_status_to_plot in [False, True]:
            color = "black" if vanished_status_to_plot else "blue"
            label = "Vanished" if vanished_status_to_plot else "Non-vanished"
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=5, label=label)
            )

        # Add the legend to the first subplot or the entire figure
        fig.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            title=None,
        )

    for ax in axes.flatten()[len(columns) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def plot_latent_dims_in_umap(
    embed: AnnData,
    title_col: str = "title",
    additional_columns: Sequence[str] = (),
    max_cells_to_plot: int | None = None,
    order_col: str = "order",
    dim_subset: Sequence[str] | None = None,
    directional: bool = False,
    remove_vanished: bool = True,
    rearrange_titles: bool = True,
    color_bar_rescale_ratio: float = 1.0,
    show: bool = True,
    **kwargs,
):
    """Plot the latent dimensions on a UMAP embedding.

    This function creates UMAP plots for each latent dimension, showing how cells
    are distributed in the UMAP space based on their values for each dimension.
    It can optionally handle directional dimensions and subsample cells for performance.

    Parameters
    ----------
    embed
        Annotated data object containing the UMAP embedding in `.obsm['X_umap']`
        and latent dimensions in `.X`.
    title_col
        Name of the column in `embed.var` to use as titles for each dimension.
        If None, default titles will be used.
    additional_columns
        Additional columns from `embed.obs` to plot alongside the latent dimensions.
    max_cells_to_plot
        Maximum number of cells to plot. If the number of cells in `embed`
        is greater than this value, a subsample will be taken.
    order_col
        The column in `embed.var` to use for ordering the dimensions.
        Ignored if `dim_subset` is provided.
    dim_subset
        The subset of dimensions to plot. If provided, overrides `order_col`.
    directional
        Whether to consider positive and negative directions as separate dimensions.
        If True, creates separate plots for + and - directions.
    remove_vanished
        Whether to remove vanished dimensions from the plot.
    rearrange_titles
        Whether to rearrange titles to the bottom right of each plot.
    color_bar_rescale_ratio
        Ratio to rescale the height of colorbars.
    show
        Whether to display the plot. If False, returns the figure object.
    **kwargs
        Additional keyword arguments passed to `sc.pl.umap`.

    Returns
    -------
    matplotlib.figure.Figure or None
        The UMAP plot figure if `show=False`, otherwise None.

    Raises
    ------
    ValueError
        If required columns (`order_col` or "vanished") are not found in `embed.var`.

    Notes
    -----
    The function expects the following columns in `embed.var`:
    - `order_col`: For ordering dimensions
    - `title_col`: For dimension titles
    - `vanished`: Boolean indicating vanished dimensions (if `remove_vanished=True`)
    - `min`, `max`: For setting color scale limits

    When `directional=True`, the function creates separate plots for positive
    and negative directions, effectively doubling the number of plots.

    Examples
    --------
    >>> # Basic UMAP plot of latent dimensions
    >>> plot_latent_dims_in_umap(embed)
    >>> # Plot with directional dimensions and custom subset
    >>> plot_latent_dims_in_umap(embed, directional=True, dim_subset=["DR 1", "DR 2"])
    >>> # Plot with additional metadata columns
    >>> plot_latent_dims_in_umap(embed, additional_columns=["cell_type", "batch"])
    """
    if order_col not in embed.var:
        raise ValueError(
            f'Column "{order_col}" not found in `embed.var`. Please run `set_latent_dimension_stats` to set order.'
        )
    if remove_vanished and "vanished" not in embed.var:
        raise ValueError(
            'Column "vanished" not found in `embed.var`. Please run `set_latent_dimension_stats` to set vanished status.'
        )

    if max_cells_to_plot is not None and embed.n_obs > max_cells_to_plot:
        embed = sc.pp.subsample(embed, n_obs=max_cells_to_plot, copy=True)

    if directional:
        embed_pos = embed.copy()
        if "layer" in kwargs:
            embed_pos.X = embed_pos.layers[kwargs["layer"]]
            del kwargs["layer"]
        embed_neg = embed_pos.copy()
        embed_neg.X = -embed_neg.X
        embed_neg.var["min"], embed_neg.var["max"] = -embed_neg.var["max"], -embed_neg.var["min"]

        embed_pos.var["_direction"] = "+"
        embed_neg.var["_direction"] = "-"
        embed_pos.var[title_col] = embed_pos.var[title_col] + "+"
        embed_neg.var[title_col] = embed_neg.var[title_col] + "-"
        if embed.var[order_col].dtype in [np.int32, np.int64, np.float32, np.float64]:
            embed_pos.var[order_col] = embed_pos.var[order_col] + 1e-8
            embed_neg.var[order_col] = embed_neg.var[order_col] - 1e-8
        else:
            embed_pos.var[order_col] = embed_pos.var[order_col].astype(str) + "+"
            embed_neg.var[order_col] = embed_neg.var[order_col].astype(str) + "-"

        embed = ad.concat([embed_pos, embed_neg], axis=1, join="inner", merge="first")
        embed.var.reset_index(drop=True, inplace=True)

    tmp_df = embed.var.sort_values(order_col)
    if remove_vanished:
        tmp_df = tmp_df.query("vanished == False")
    if dim_subset:
        tmp_df = tmp_df.set_index(title_col).loc[dim_subset].reset_index()
    cols_to_show = tmp_df.index if title_col is None else tmp_df[title_col]
    if additional_columns:
        cols_to_show = list(cols_to_show) + list(additional_columns)

    kwargs = {
        **dict(  # noqa: C408
            frameon=False,
            cmap=cmap.saturated_red_blue_cmap if not directional else cmap.saturated_sky_cmap,
            vmin=list(np.minimum(tmp_df["min"].values, -1)),
            vcenter=0,
            vmax=list(np.maximum(tmp_df["max"].values, +1)),
        ),
        **kwargs,
    }
    fig = sc.pl.umap(embed, gene_symbols=title_col, color=cols_to_show, return_fig=True, **kwargs)
    for i, ax in enumerate(fig.axes[1 : 2 * len(tmp_df) : 2]):
        assert hasattr(ax, "_colorbar")
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0, pos.width, pos.height * color_bar_rescale_ratio]
        ax.set_position(new_pos)

        if directional:
            direction = tmp_df["_direction"].iloc[i]
            if direction == "-":
                ax.invert_yaxis()
                labels = -ax.get_yticks()
                if all(x == int(x) for x in labels):
                    labels = [int(x) for x in labels]
                ax.set_yticklabels(labels)
    if rearrange_titles:
        for ax in fig.axes:
            ax.text(0.935, 0.05, ax.get_title(), size=15, ha="left", color="black", rotation=90, transform=ax.transAxes)
            ax.set_title("")

    if show:
        plt.show()
    else:
        return fig


def plot_latent_dims_in_heatmap(
    embed: AnnData,
    categorical_column: str,
    title_col: str | None = "title",
    sort_by_categorical: bool = False,
    make_balanced: bool = True,
    order_col: str | None = "order",
    remove_vanished: bool = True,
    figsize: tuple[int, int] | None = None,
    show: bool = True,
    **kwargs,
):
    """Plot the latent dimensions in a heatmap.

    This function creates a heatmap showing the values of latent dimensions
    across different categories. It can optionally create balanced subsamples
    and sort dimensions based on categorical differences.

    Parameters
    ----------
    embed
        Annotated data object containing the latent dimensions in `.X`
        and categorical metadata in `.obs`.
    categorical_column
        The column in `embed.obs` that represents the categorical variable
        for grouping cells.
    title_col
        The column in `embed.var` to use as titles for each dimension.
        If None, uses the dimension indices.
    sort_by_categorical
        Whether to sort dimensions based on their maximum absolute values
        within each category. If True, `order_col` is ignored.
    make_balanced
        Whether to create a balanced subsample of the data based on the
        categorical variable using `make_balanced_subsample`.
    order_col
        The column in `embed.var` to use for ordering the dimensions.
        Ignored if `sort_by_categorical=True`.
    remove_vanished
        Whether to remove vanished dimensions from the plot.
    figsize
        The size of the figure (width, height) in inches.
        If None, automatically calculated based on number of categories.
    show
        Whether to display the plot. If False, returns the plot object.
    **kwargs
        Additional keyword arguments passed to `sc.pl.heatmap`.

    Returns
    -------
    matplotlib.axes.Axes or None
        The heatmap axes if `show=False`, otherwise None.

    Raises
    ------
    ValueError
        If required columns (`order_col` or "vanished") are not found in `embed.var`.

    Notes
    -----
    The function expects the following columns in `embed.var`:
    - `order_col`: For ordering dimensions (if `sort_by_categorical=False`)
    - `title_col`: For dimension titles
    - `vanished`: Boolean indicating vanished dimensions (if `remove_vanished=True`)

    If `figsize=None`, the figure height is automatically calculated as
    `len(unique_categories) / 6` to accommodate all categories.

    The heatmap uses a red-blue color map centered at 0, with no dendrogram.

    Examples
    --------
    >>> # Basic heatmap of latent dimensions by cell type
    >>> plot_latent_dims_in_heatmap(embed, categorical_column="cell_type")
    >>> # Heatmap with balanced sampling and custom sorting
    >>> plot_latent_dims_in_heatmap(embed, categorical_column="condition", sort_by_categorical=True, make_balanced=True)
    >>> # Heatmap with custom figure size
    >>> plot_latent_dims_in_heatmap(embed, categorical_column="batch", figsize=(12, 8))
    """
    if order_col is not None and order_col not in embed.var:
        raise ValueError(
            f'Column "{order_col}" not found in `embed.var`. Please run `set_latent_dimension_stats` to set order.'
        )
    if remove_vanished:
        if "vanished" not in embed.var:
            raise ValueError(
                'Column "vanished" not found in `embed.var`. Please run `set_latent_dimension_stats` to set vanished status.'
            )
        embed = embed[:, ~embed.var["vanished"]]

    if make_balanced:
        embed = make_balanced_subsample(embed, categorical_column)

    if sort_by_categorical:
        dim_order = np.abs(embed.X).argmax(axis=0).argsort().tolist()
    elif order_col is None:
        dim_order = np.arange(embed.n_vars)
    else:
        dim_order = embed.var[order_col].argsort().tolist()

    if title_col is None:
        vars_to_show = embed.var.iloc[dim_order].index
    else:
        vars_to_show = embed.var.iloc[dim_order][title_col]

    if figsize is None:
        figsize = (10, len(embed.obs[categorical_column].unique()) / 6)

    kwargs = {
        **dict(  # noqa: C408
            vcenter=0,
            cmap=cmap.saturated_red_blue_cmap,
            dendrogram=False,
        ),
        **kwargs,
    }

    return sc.pl.heatmap(
        embed,
        vars_to_show,
        categorical_column,
        gene_symbols=title_col,
        figsize=figsize,
        show_gene_labels=True,
        show=show,
        **kwargs,
    )
