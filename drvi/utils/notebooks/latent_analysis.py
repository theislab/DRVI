import anndata as ad
import numpy as np
import scanpy as sc


def prepare_pca(adata, pca_calc=None, **kwargs):
    if pca_calc is None:
        pca_calc = 'pca' not in adata.uns
    if pca_calc:
        sc.tl.pca(adata, **kwargs)


def plot_pca(adata, color=None, pca_calc=None, pca_kwargs={}, **kwargs):
    prepare_pca(adata, pca_calc=pca_calc, **pca_kwargs)
    return sc.pl.pca(adata, color=color, **kwargs)


def plot_pca_variance_ratio(adata, pca_calc=None, pca_kwargs={}, **kwargs):
    prepare_pca(adata, pca_calc=pca_calc, **pca_kwargs)
    return sc.pl.pca_variance_ratio(adata, log=True, **kwargs)


def prepare_neighborhood_graph(adata, ng_calc=None, **kwargs):
    if ng_calc is None:
        ng_calc = 'neighbors' not in adata.uns
    if ng_calc:
        sc.pp.neighbors(adata, **{'n_neighbors': 10, 'n_pcs': 20, **kwargs})


def prepare_umap(adata, pca_calc=None, pca_kwargs={}, ng_calc=None, ng_kwargs={}, umap_calc=None,
                 random_state=123, **kwargs):
    if umap_calc is None:
        umap_calc = 'umap' not in adata.uns
    if umap_calc:
        prepare_pca(adata, pca_calc=pca_calc, **pca_kwargs)
        prepare_neighborhood_graph(adata, ng_calc, **ng_kwargs)
        sc.tl.umap(adata, **{'spread': 1.0, 'min_dist': 0.5, 'random_state': random_state, **kwargs})


def plot_umap(adata, color=None, pca_calc=None, pca_kwargs={}, ng_calc=None, ng_kwargs={}, umap_calc=None,
              umap_kwargs={}, **kwargs):
    prepare_umap(adata, pca_calc, pca_kwargs, ng_calc, ng_kwargs, umap_calc, **umap_kwargs)
    return sc.pl.umap(adata, color=color, **kwargs)


def cluster_cells(adata, ng_calc=None, ng_kwargs={}, resolution=0.5):
    prepare_neighborhood_graph(adata, ng_calc, **ng_kwargs)
    sc.tl.leiden(adata, resolution=resolution)


# From pertpy
def sorted_heatmap(
    adata: ad.AnnData,
    layer=None,
    order_by=None,
    key_to_save_order=None,
    groupby=None,
    **kwds,
):
    data = adata.X if layer is None else adata.layers[layer]

    if order_by is None:
        max_guide_index = np.where(
            np.array(data.max(axis=1)).squeeze() != data.min(), np.array(data.argmax(axis=1)).squeeze(), -1
        )
        order = np.argsort(max_guide_index)
    elif isinstance(order_by, str):
        order = adata.obs[order_by]
    else:
        order = order_by

    if groupby is None:
        adata.obs["_tmp_pertpy_grna_plot_dummy_group"] = ""
    else:
        adata.obs["_tmp_pertpy_grna_plot_dummy_group"] = adata.obs[groupby]
    if key_to_save_order is not None:
        adata.obs[key_to_save_order] = order
    axis_group = sc.pl.heatmap(
        adata[order],
        adata.var.index.tolist(),
        groupby="_tmp_pertpy_grna_plot_dummy_group",
        cmap="viridis",
        use_raw=False,
        dendrogram=False,
        layer=layer,
        **kwds,
    )
    del adata.obs["_tmp_pertpy_grna_plot_dummy_group"]
    return axis_group
