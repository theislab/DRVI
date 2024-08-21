import anndata as ad
import numpy as np
import scanpy as sc


def plot_latent_dims_in_umap(embed_adata, dims=None, max_cells_to_plot=None, optimal_order=False, **kwargs):
    if max_cells_to_plot is not None and embed_adata.n_obs > max_cells_to_plot:
        embed_adata = sc.pp.subsample(embed_adata, n_obs=max_cells_to_plot, copy=True)
    if dims is None:
        dims = list(range(embed_adata.n_vars))
    obs_original = embed_adata.obs.copy()
    for i in dims:
        embed_adata.obs[f'Dim {i+1}'] = embed_adata.X[:, i]
    try:
        if optimal_order:
            color_cols = [f'Dim {i+1}' for i in 
                          embed_adata.uns['optimal_var_order'] if i in dims]
        else:
            color_cols = [f'Dim {i+1}' for i in dims]
        kwargs = {**dict(
            frameon=False,
        ), **kwargs}
        pl = sc.pl.umap(embed_adata, color=color_cols, return_fig=True, **kwargs)
    finally:
        embed_adata.obs = obs_original
    return pl
