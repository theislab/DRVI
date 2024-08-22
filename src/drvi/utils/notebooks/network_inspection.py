import anndata as ad
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy


def _find_optimal_ordering(node_activity):
    X = node_activity.T
    Z = hierarchy.linkage(X)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, X))


def plot_group_activity(node_activities, groups):
    for group_name, group_series in groups.items():
        print(f"Plotting for {group_name}")
        for group_value in group_series.unique():
            for _name, node_activity in node_activities.items():
                df = pd.DataFrame(
                    {
                        **{f"{i}": node_activity[:, i] for i in range(node_activity.shape[1])},
                        f"is_{group_value}": group_series == group_value,
                    }
                )
                df = pd.melt(df, id_vars=[f"is_{group_value}"], var_name="node", value_name="value")
                fig, ax = plt.subplots(1, 1, figsize=(80, 10))
                oo = [f"{i}" for i in _find_optimal_ordering(node_activity)]
                sns.violinplot(data=df, x="node", y="value", order=oo, hue=f"is_{group_value}", split=True, ax=ax)
                fig.show()
                plt.show()


def plot_group_activity_matrix(
    node_activities, groups, dendrogram=True, figsize=(10, 5), violin_plot=True, cmap="Blues", vcenter=0
):
    n_plots = (2 if violin_plot else 1) * len(node_activities) * len(groups)
    fig, axes = plt.subplots(n_plots, 1, figsize=(figsize[0], figsize[1] * n_plots), squeeze=False)
    i = -1
    for name, node_activity in node_activities.items():
        adata = ad.AnnData(
            node_activity,
            var=pd.DataFrame(index=[f"{i}" for i in range(node_activity.shape[1])]),
            obs=pd.DataFrame(groups),
        )
        oo = _find_optimal_ordering(node_activity)
        for group_name in groups.keys():
            i += 1
            ax = axes[i, 0]
            sc.pl.matrixplot(
                adata,
                [col for col in adata.var.index[oo] if col not in [group_name]],
                groupby=group_name,
                dendrogram=dendrogram,
                cmap=cmap,
                ax=ax,
                show=False,
                vcenter=vcenter,
            )
            ax.set_title(f"Matrix_{name}_{group_name}")

            if violin_plot:
                i += 1
                ax = axes[i, 0]
                sc.pl.stacked_violin(
                    adata,
                    [col for col in adata.var.index[oo] if col not in [group_name]],
                    groupby=group_name,
                    swap_axes=False,
                    dendrogram=dendrogram,
                    standard_scale="var",
                    ax=ax,
                    show=False,
                )
                ax.set_title(f"Violin_{name}_{group_name}")

    return fig
