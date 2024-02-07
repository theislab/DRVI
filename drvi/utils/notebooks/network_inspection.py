import anndata as ad
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy

from .latent_analysis import plot_umap


def plot_node_activity(node_activities, groups, umap_colors=None, umap_adata=None, max_output=5):
    any_node_activity = list(node_activities.values())[0]
    
    for i in range(min(max_output, any_node_activity.shape[1])):
        df = pd.DataFrame({**{name: value[:, i] for name, value in node_activities.items()},
                           **groups})
        for node_activity_key in node_activities.keys():
            fig, ax = plt.subplots(1, len(groups), figsize=(10 * len(groups), 10))
            for j, group_name in enumerate(groups.keys()):
                sns.violinplot(x=group_name, y=node_activity_key, data=df, ax=ax[j])
                ax[j].tick_params(axis='x', rotation=90)
            fig.show()
            plt.show()
        if umap_colors is not None:
            colors = []
            for umap_color in umap_colors:
                if umap_color in df.columns:
                    umap_adata.obs[f'_tmp_{umap_color}'] = df[umap_color]
                    colors.append(f'_tmp_{umap_color}')
                else:
                    colors.append(umap_color)
            with plt.rc_context({"figure.figsize": (10, 10), 'figure.dpi': 40}):
                plot_umap(umap_adata, color=colors)
                plt.show()
            for umap_color in umap_colors:
                col = f'_tmp_{umap_color}'
                if col in umap_adata.obs.columns:
                    umap_adata.obs.drop(columns=[col], inplace=True)


def _find_optimal_ordering(node_activity):
    X = node_activity.T
    Z = hierarchy.linkage(X)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))


def plot_group_activity(node_activities, groups):
    for group_name, group_series in groups.items():
        print(f"Plotting for {group_name}")
        for group_value in group_series.unique():
            for name, node_activity in node_activities.items():
                df = pd.DataFrame({**{f"{i}": node_activity[:, i] for i in range(node_activity.shape[1])},
                                   f'is_{group_value}': group_series == group_value})
                df = pd.melt(df, id_vars=[f'is_{group_value}'], var_name='node', value_name='value')
                fig, ax = plt.subplots(1, 1, figsize=(80, 10))
                oo = [f"{i}" for i in _find_optimal_ordering(node_activity)]
                sns.violinplot(data=df, x='node', y='value', order=oo, hue=f'is_{group_value}', split=True, ax=ax)
                fig.show()
                plt.show()


def plot_group_activity_matrix(node_activities, groups, dendrogram=True, figsize=(10, 5), violin_plot=True, cmap='Blues', vcenter=0):
    n_plots = (2 if violin_plot else 1) * len(node_activities) * len(groups)
    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    i = -1
    for name, node_activity in node_activities.items():
        adata = ad.AnnData(node_activity, 
                           var=pd.DataFrame(index=[f"{i}" for i in range(node_activity.shape[1])]),
                           obs=pd.DataFrame(groups),)
        oo = _find_optimal_ordering(node_activity)
        for group_name in groups.keys():
            i += 1
            ax = axes[i, 0]
            sc.pl.matrixplot(adata, [col for col in adata.var.index[oo] if col not in [group_name]], 
                             groupby=group_name, dendrogram=dendrogram,
                             cmap=cmap, ax=ax, show=False, vcenter=vcenter)
            ax.set_title(f"Matrix_{name}_{group_name}")
            
            if violin_plot:
                i += 1
                ax = axes[i, 0]
                sc.pl.stacked_violin(adata, [col for col in adata.var.index[oo] if col not in [group_name]], 
                                    groupby=group_name, swap_axes=False, dendrogram=dendrogram,
                                    standard_scale='var', ax=ax, show=False)
                ax.set_title(f"Violin_{name}_{group_name}")
            
    return fig
