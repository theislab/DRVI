from collections.abc import Sequence
from typing import Literal

import numpy as np
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt

from drvi.utils.plotting import cmap


def make_balanced_subsample(adata, col, min_count=10):
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
    titles: dict[str, str] = None,
    remove_vanished: bool = False,
    show: bool = True,
):
    """
    Plot the statistics of latent dimensions.

    Args:
    - embed (AnnData): The annotated data object containing the latent dimensions.
    - figsize (Tuple[int, int], optional): The size of the figure (width, height). Default is (5, 3).
    - log_scale (Union[bool, Literal['try']], optional): Whether to use a log scale for the y-axis. If 'try', the log scale is used if the minimum value is greater than 0. Default is 'try'.
    - ncols (int, optional): The maximum number of columns in the subplot grid. Default is 5.
    - columns (Sequence[str], optional): The columns to plot from the `embed` object. Default is ('reconstruction_effect', 'max_value', 'mean', 'std').
    - titles (Dict[str, str], optional): The titles for each column in the plot.
    - show (bool, optional): Whether to display the plot. If False, figure is returned. Default is True.

    Returns
    -------
    - plt (matplotlib.pyplot module): The matplotlib.pyplot module if show is False.
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
    embed,
    title_col: str = "title",
    additional_columns=(),
    max_cells_to_plot: int | None = None,
    order_col="order",
    remove_vanished: bool = True,
    show: bool = True,
    **kwargs,
):
    """
    Plot the latent dimensions of a UMAP embedding.

    Args:
        embed (AnnData): Annotated data object containing the UMAP embedding.
        title_col (str, optional): Name of the column in `embed.var` to use as titles for each dimension.
                                   If None, default titles will be used.
        additional_columns (tuple, optional): Additional columns to plot alongside the latent dimensions.
        max_cells_to_plot (int, optional): Maximum number of cells to plot. If the number of cells in `embed`
                                           is greater than `max_cells_to_plot`, a subsample will be taken.
        order_col (str, optional): The column in the `embed.var` DataFrame to use for ordering the dimensions.
        remove_vanished (bool, optional): Whether to remove the vanished dimensions from the plot. Default is True.
        show (bool, optional): Whether to display the plot. If False, the plot is returned. Default is True.
        **kwargs: Additional keyword arguments to be passed to `sc.pl.umap`.

    Returns
    -------
        matplotlib.figure.Figure: The UMAP plot figure if show is False.
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

    tmp_df = embed.var.sort_values(order_col)
    if remove_vanished:
        tmp_df = tmp_df.query("vanished == False")
    cols_to_show = tmp_df.index if title_col is None else tmp_df[title_col]
    if additional_columns:
        cols_to_show = list(cols_to_show) + list(additional_columns)

    kwargs = {
        **dict(  # noqa: C408
            frameon=False,
            cmap=cmap.saturated_red_blue_cmap,
            vmin=list(np.minimum(tmp_df["min"].values, -1)),
            vcenter=0,
            vmax=list(np.maximum(tmp_df["max"].values, +1)),
        ),
        **kwargs,
    }
    pl = sc.pl.umap(embed, gene_symbols=title_col, color=cols_to_show, return_fig=True, **kwargs)

    if show:
        plt.show()
    else:
        return pl


def plot_latent_dims_in_heatmap(
    embed: AnnData,
    categorical_column: str,
    title_col: str | None = None,
    sort_by_categorical: bool = False,
    make_balanced: bool = True,
    order_col: str | None = "order",
    remove_vanished: bool = True,
    figsize: tuple[int, int] | None = None,
    show: bool = True,
    **kwargs,
):
    """
    Plot the latent dimensions in a heatmap.

    Parameters
    ----------
        embed (AnnData): The annotated data object containing the latent dimensions.
        categorical_column (str): The column in the `embed.obs` DataFrame that represents the categorical variable.
        title_col (str, optional): The column in the `embed.var` DataFrame to use as the title for each dimension. Defaults to None.
        sort_by_categorical (bool, optional): Whether to sort the dimensions based on the categorical variable. Defaults to True.
        make_balanced (bool, optional): Whether to make a balanced subsample of the data based on the categorical variable. Defaults to True.
        order_col (str, optional): The column in the `embed.var` DataFrame to use for ordering the dimensions. Discarded if sort_by_categorical is set. Defaults to 'order'.
        remove_vanished (bool, optional): Whether to remove the vanished dimensions from the plot. Default is True.
        figsize (Tuple[int, int], optional): The size of the figure. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the `sc.pl.heatmap` function.

    Returns
    -------
        plot if Show is not False.
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
