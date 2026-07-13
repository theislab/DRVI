from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def plot_interpretability_scores(
    gene_scores_df: pd.DataFrame,
    n_top_genes: int = 10,
    ncols: int = 5,
    score_threshold: float = 0.1,
    dim_subset: Sequence[str] | None = None,
    show: bool = True,
    **kwargs,
):
    """Plot bar plots of the top genes per latent dimension for an interpretability score.

    This function visualizes the result of
    :meth:`~DRVI.get_interpretability_scores`.
    For each latent dimension (a column of ``gene_scores_df``) it draws a horizontal bar plot
    of the ``n_top_genes`` genes with the highest scores.

    Parameters
    ----------
    gene_scores_df
        DataFrame of interpretability scores with genes as rows and latent dimensions as
        columns, as returned by ``get_interpretability_scores``.
    n_top_genes
        Number of top genes to display per dimension.
    ncols
        Number of columns in the subplot grid.
    score_threshold
        Minimum score for a dimension to be plotted. Dimensions whose maximum gene score is
        below this threshold are skipped.
    dim_subset
        Subset of dimension titles (column names) to plot. If None, all dimensions meeting
        ``score_threshold`` are plotted, ordered by their dimension number.
    show
        Whether to display the plot. If False, returns the figure object.
    **kwargs
        Additional keyword arguments passed to :meth:`matplotlib.axes.Axes.barh`.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if ``show=False``, otherwise None.
    """
    info = {k: v for k, v in gene_scores_df.to_dict(orient="series").items() if v.max() >= score_threshold}
    if dim_subset is None:
        dims = sorted(info, key=lambda x: int(re.search(r"\d+", x).group()))
    else:
        dims = [dim for dim in dim_subset if dim in info]

    n_row = int(np.ceil(len(dims) / ncols))
    fig, axes = plt.subplots(n_row, ncols, figsize=(3 * ncols, int(1 + 0.2 * n_top_genes) * n_row))
    axes = np.atleast_1d(axes).flatten()

    barh_kwargs = {"color": "skyblue", **kwargs}
    for ax, dim in zip(axes, dims, strict=False):
        top = info[dim].sort_values(ascending=False)[:n_top_genes]
        ax.barh(top.index, top.values, **barh_kwargs)
        ax.set_title(dim)
        ax.set_xlabel("Gene score")
        ax.invert_yaxis()
        ax.grid(False)

    for ax in axes[len(dims) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig
