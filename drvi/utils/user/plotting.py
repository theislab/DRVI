import itertools
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib import pyplot as plt

from drvi.utils import cmap
from drvi.utils.interpretability import get_dimensions_of_traverse_data
from drvi.utils.user.general import (iterate_on_top_differential_vars,
                                     make_balanced_subsample)


def plot_latent_dimension_stats(
        embed: AnnData,
        figsize: Tuple[int, int] = (5, 3),
        log_scale: Union[bool, Literal['try']] = 'try',
        ncols: int = 5,
        columns: Sequence[str] = ('reconstruction_effect', 'max_value', 'mean', 'std'),
        titles: Dict[str, str] = {
            'reconstruction_effect': 'Reconstruction effect',
            'max_value': 'Max value',
            'mean': 'Mean',
            'std': 'Standard Deviation',
        },
        remove_vanished: bool=False,
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

    Returns:
    - plt (matplotlib.pyplot module): The matplotlib.pyplot module if show is False.
    """
    nrows = int(np.ceil(len(columns) / ncols))
    if nrows == 1:
        ncols = len(columns)

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows), 
                             sharey=False, sharex=False, squeeze=False)

    # Iterate through columns and plot the data
    for ax, col in zip(axes.flatten(), columns):
        df = embed.var
        if remove_vanished:
            df = df.query('vanished == False')
        df = df.sort_values('order')
        ranks = df['order']
        x = df[col]
        
        ax.plot(ranks, x, linestyle='-', color='grey', label='Line')  # Solid line plot
        for vanished_status_to_plot in [True, False]:
            indices = df['vanished'] == vanished_status_to_plot
            ax.plot(ranks[indices], x[indices], 'o', markersize=3, 
                    color='black' if vanished_status_to_plot else 'blue', label='Data Points')
        
        # Adding labels and title
        ax.set_xlabel('Rank based on Explanation Share')
        ax.set_ylabel(titles[col] if col in titles else col)
        if isinstance(log_scale, str):
            if log_scale == 'try':
                if x.min() > 0:
                    ax.set_yscale('log')
        else:
            if log_scale:
                ax.set_yscale('log')
        
        # Removing the legend
        ax.legend().remove()
        
        # Adding grid
        ax.grid(axis='x')

    if not remove_vanished:
        # Create custom legend entries
        handles, labels = [], []
        for vanished_status_to_plot in [False, True]:
            color = 'black' if vanished_status_to_plot else 'blue'
            label = 'Vanished' if vanished_status_to_plot else 'Non-vanished'
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5, label=label))
        
        # Add the legend to the first subplot or the entire figure
        fig.legend(handles=handles, labels=[handle.get_label() for handle in handles], 
            loc='center left', bbox_to_anchor=(1, 0.5), title=None)


    for ax in axes.flatten()[len(columns):]:
        fig.delaxes(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def plot_latent_dims_in_umap(
        embed, 
        title_col: str='title',
        additional_columns=tuple(),
        max_cells_to_plot: Union[int, None]=None,
        order_col='order',
        remove_vanished: bool=True,
        show: bool=True,
        **kwargs
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
    Returns:
        matplotlib.figure.Figure: The UMAP plot figure if show is False.
    """
    if order_col not in embed.var:
        raise ValueError(f'Column "{order_col}" not found in `embed.var`. Please run `set_latent_dimension_stats` to set order.')
    if remove_vanished and 'vanished' not in embed.var:
        raise ValueError('Column "vanished" not found in `embed.var`. Please run `set_latent_dimension_stats` to set vanished status.')
    
    if max_cells_to_plot is not None and embed.n_obs > max_cells_to_plot:
        embed = sc.pp.subsample(embed, n_obs=max_cells_to_plot, copy=True)
    
    tmp_df = embed.var.sort_values(order_col)
    if remove_vanished:
        tmp_df = tmp_df.query('vanished == False')
    cols_to_show = tmp_df.index if title_col is None else tmp_df[title_col]
    if additional_columns:
        cols_to_show = list(cols_to_show) + list(additional_columns)

    kwargs = {**dict(
        frameon=False,
        cmap=cmap.saturated_red_blue_cmap,
        vmin=list(np.minimum(embed.X.min(axis=0), -1)),
        vcenter=0,
        vmax=list(np.maximum(embed.X.max(axis=0), +1)),
    ), **kwargs}
    pl = sc.pl.umap(embed, gene_symbols=title_col, color=cols_to_show, return_fig=True, **kwargs)

    if show:
        plt.show()
    else:
        return pl


def plot_latent_dims_in_heatmap(
        embed: AnnData,
        categorical_column: str,
        title_col: Optional[str] = None,
        sort_by_categorical: bool = False,
        make_balanced: bool = True,
        order_col: Union[str, None]='order',
        remove_vanished: bool=True,
        figsize: Optional[Tuple[int, int]] = None,
        show: bool = True,
        **kwargs,
    ):
    """
    Plot the latent dimensions in a heatmap.

    Parameters:
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

    Returns:
        plot if Show is not False.
    """
    if order_col is not None and order_col not in embed.var:
        raise ValueError(f'Column "{order_col}" not found in `embed.var`. Please run `set_latent_dimension_stats` to set order.')
    if remove_vanished:
        if 'vanished' not in embed.var:
            raise ValueError('Column "vanished" not found in `embed.var`. Please run `set_latent_dimension_stats` to set vanished status.')
        embed = embed[:, ~embed.var['vanished']]
    
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

    kwargs = {**dict(
            vcenter=0,
            cmap=cmap.saturated_red_blue_cmap,
            dendrogram=False,
        ), **kwargs}
    
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


def make_heatmap_groups(ordered_list):
    n_groups, group_names = zip(*[(len(list(group)), key) for (key, group) in itertools.groupby(ordered_list)])
    group_positions = [0] + list(itertools.accumulate(n_groups))
    group_positions = list(zip(group_positions[:-1], [c - 1 for c in group_positions[1:]]))
    return group_positions, group_names


def differential_vars_heatmap(
        traverse_adata: AnnData,
        key: str,
        title_col: str = 'title',
        score_threshold: float = 0.,
        remove_vanished: bool = True,
        remove_unaffected: bool = False,
        figsize: Optional[Tuple[int, int]] = None,
        show: bool=True,
    ):
    """
    Generate a heatmap of differential variables based on traverse data.

    Parameters:
    - traverse_adata (AnnData): Annotated data object containing traverse data.
    - key (str): Key used to access traverse effect data in `traverse_adata.varm`.
    - title_col (str): Column name in `traverse_adata.obs` to use as dimension labels.
    - score_threshold (float): Threshold value for filtering variables based on the score.
    - remove_vanished (bool): Whether to remove variables that have vanished.
    - remove_unaffected (bool): Whether to remove variables that have no effect.
    - figsize (Optional[Tuple[int, int]]): Size of the figure (width, height).
    - show (bool): Whether to show the heatmap.

    Returns:
    - None if show is True, otherwise the plot.
    """
    n_latent, n_steps, n_samples, n_vars = get_dimensions_of_traverse_data(traverse_adata)
    
    max_effect_index_in_positive_direction_for_each_gene = np.abs(traverse_adata.varm[f"{key}_traverse_effect_pos"]).values.argmax(axis=1)
    max_effect_in_positive_direction_for_each_gene = np.abs(traverse_adata.varm[f"{key}_traverse_effect_pos"].values[np.arange(traverse_adata.n_vars), max_effect_index_in_positive_direction_for_each_gene])
    max_effect_index_in_negative_direction_for_each_gene = np.abs(traverse_adata.varm[f"{key}_traverse_effect_neg"]).values.argmax(axis=1)
    max_effect_in_negative_direction_for_each_gene = np.abs(traverse_adata.varm[f"{key}_traverse_effect_neg"].values[np.arange(traverse_adata.n_vars), max_effect_index_in_negative_direction_for_each_gene])
    traverse_adata.var[f'max_effect'] = np.maximum(max_effect_in_positive_direction_for_each_gene, max_effect_in_negative_direction_for_each_gene)
    for col in ['dim_id', 'order', title_col]:
        title_mapping = dict(zip(traverse_adata.obs['dim_id'].values, traverse_adata.obs[col].values))
        traverse_adata.var[f'max_effect_dim_{col}'] = np.where(
            traverse_adata.var[f'max_effect'] < score_threshold,
            float('nan') if np.isreal(traverse_adata.obs[col].values[0]) else 'NONE',
            np.where(
                max_effect_in_positive_direction_for_each_gene > max_effect_in_negative_direction_for_each_gene,
                pd.Series(max_effect_index_in_positive_direction_for_each_gene).map(title_mapping),
                pd.Series(max_effect_index_in_negative_direction_for_each_gene).map(title_mapping),
            )
        )
        traverse_adata.var[f'max_effect_dim_{col}_plus'] = np.where(
            traverse_adata.var[f'max_effect'] < score_threshold,
            'NONE',
            np.where(
                max_effect_in_positive_direction_for_each_gene > max_effect_in_negative_direction_for_each_gene,
                pd.Series(max_effect_index_in_positive_direction_for_each_gene).map(title_mapping).astype(str) + ' +',
                pd.Series(max_effect_index_in_negative_direction_for_each_gene).map(title_mapping).astype(str) + ' -',
            )
        )

    plot_adata = AnnData(
        traverse_adata.uns[f"{key}_traverse_effect_stepwise"].reshape(n_latent * n_steps, n_vars),
        var=traverse_adata.var,
        obs=pd.DataFrame({
            'dim_id': np.repeat(np.arange(n_latent), n_steps),
            'step_id': np.tile(np.arange(n_steps), n_latent),
        }),
    )
    for col in ['dim_id', 'order', title_col, 'vanished']:
        title_mapping = dict(zip(traverse_adata.obs['dim_id'].values, traverse_adata.obs[col].values))
        plot_adata.obs[col] = plot_adata.obs['dim_id'].map(title_mapping)

    if remove_vanished:
        plot_adata = plot_adata[~plot_adata.obs['vanished']].copy()
    if remove_unaffected:
        plot_adata = plot_adata[:, plot_adata.var['max_effect'] > score_threshold].copy()
    plot_adata = plot_adata[:, plot_adata.var.sort_values(["max_effect_dim_order", "max_effect_dim_order_plus", "max_effect"]).index].copy()
    plot_adata = plot_adata[plot_adata.obs.sort_values(["order"]).index].copy()

    if figsize is None:
        figsize = (20, plot_adata.obs['dim_id'].nunique() / 4)

    vmin = min(-1, min(traverse_adata.varm[f"{key}_traverse_effect_pos"].values.min(), traverse_adata.varm[f"{key}_traverse_effect_neg"].values.min()))
    vmax = max(+1, max(traverse_adata.varm[f"{key}_traverse_effect_pos"].values.max(), traverse_adata.varm[f"{key}_traverse_effect_neg"].values.max()))
    var_group_positions, var_group_labels = make_heatmap_groups(plot_adata.var[f'max_effect_dim_{title_col}_plus'])

    return sc.pl.heatmap(
        plot_adata,
        plot_adata.var.index,
        groupby=title_col,
        layer=None,
        figsize=figsize,
        dendrogram=False,
        vcenter=0, vmin=vmin, vmax=vmax,
        cmap=cmap.saturated_red_blue_cmap,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        var_group_rotation=90,
        show=show,
    )


def _bar_plot_top_differential_vars(
        plot_info: Sequence[Tuple[str, pd.Series]],
        ncols: int = 5,
        show: bool = True,
    ):
    """
    Plot the top differential variables in a bar plot.

    Parameters:
        plot_info (Sequence[Tuple[str, pd.Series]]): Information about the top differential variables.
        ncols (int, optional): Number of columns in the plot grid. Defaults to 5.
        show (bool, optional): Whether to display the plot. If False, the plot will be returned as a Figure object. Defaults to True.
    
    Returns:
        None if show is True, otherwise the figure.
    """
    n_row = int(np.ceil(len(plot_info) / ncols))
    fig, axes = plt.subplots(n_row, ncols, figsize=(3 * ncols, 3 * n_row))

    for ax, info in zip(axes.flatten(), plot_info):
        dim_title = info[0]
    
        top_indices = info[1].sort_values(ascending=False)[:10]
        genes = top_indices.index
        values = top_indices.values

        # Create a horizontal bar plot
        ax.barh(genes, values, color='skyblue')
        ax.set_xlabel('Gene Score')
        ax.set_title(dim_title)
        ax.invert_yaxis()
        ax.grid(False)

    for ax in axes.flatten()[len(plot_info):]:
        fig.delaxes(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig




def show_top_differential_vars(
        traverse_adata: AnnData,
        key: str,
        title_col: str = 'title',
        order_col: str = 'order',
        gene_symbols: Optional[str] = None,
        score_threshold: float = 0.,
        ncols: int = 5,
        show: bool = True,
    ):
    """
    Show top differential variables in a bar plot.

    Parameters:
        traverse_adata (AnnData): Annotated data object containing the variables to be plotted.
        key (str): Key to access the traverse effect variables in `traverse_adata.varm`.
        title_col (str, optional): Column name in `traverse_adata.obs` that contains the titles for each dimension. Defaults to 'title'.
        order_col (str, optional): Column name in `traverse_adata.obs` that specifies the order of dimensions. Defaults to 'order'.
        gene_symbols (str, optional): Column name in `traverse_adata.var` that contains gene symbols. If provided, gene symbols will be used in the plot instead of gene indices. Defaults to None.
        score_threshold (float, optional): Threshold value for gene scores. Only genes with scores above this threshold will be plotted. Defaults to 0.
        ncols (int, optional): Number of columns in the plot grid. Defaults to 5.
        show (bool, optional): Whether to display the plot. If False, the plot will be returned as a Figure object. Defaults to True.

    Returns:
        None if show is True, otherwise the figure.
    """
    
    plot_info = iterate_on_top_differential_vars(
        traverse_adata, key, title_col, order_col, gene_symbols, score_threshold
    )

    return _bar_plot_top_differential_vars(plot_info, ncols, show)
