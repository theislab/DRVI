# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: drvi
#     language: python
#     name: drvi
# ---

# # Imports

# %load_ext autoreload
# %autoreload 2

# +
# TODO: uncomment for the final version
# import warnings
# warnings.filterwarnings('ignore')

# +
import anndata as ad
import scanpy as sc
import numpy as np

import drvi
from drvi import DRVI
from drvi.utils.hvg import hvg_batch

from matplotlib import pyplot as plt
# -

sc.settings.set_figure_params(dpi=100, frameon=False)
sc.set_figure_params(dpi=100)
sc.set_figure_params(figsize=(3, 3))
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (3, 3)

# # Load Data

# +
# # !mkdir tmp
# # !wget -O tmp/immune_all.h5ad https://figshare.com/ndownloader/files/25717328
# -

adata = sc.read('tmp/immune_all.h5ad')
# Remove dataset with non-count values
adata = adata[adata.obs['batch'] != 'Villani'].copy()
adata

# # Pre-processing

adata.X = adata.layers['counts'].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata

# +
# sc.pp.pca(adata)
# sc.pp.neighbors(adata)
# sc.tl.umap(adata)
# adata
# -

# Batch aware HVG selection (method is obtained from scIB metrics)
hvg_genes = hvg_batch(adata, batch_key='batch', target_genes=2000, adataOut=False)
adata = adata[:, hvg_genes].copy()
adata



sc.pl.umap(adata, color=['batch', 'final_annotation'], ncols=1, frameon=False)



# # Train DRVI

# +
# Setup data
DRVI.setup_anndata(
    adata,
    # DRVI accepts count data by default.
    # Do not forget to change gene_likelihood if you provide a non-count data.
    layer='counts',
    # Always provide a list. DRVI can accept multiple covariates.
    categorical_covariate_keys=['batch'],
    # DRVI accepts count data by default.
    # Set to false if you provide log-normalized data and use normal distribution (mse loss).
    is_count_data=True,
)

# construct the model
model = DRVI(
    adata,
    # Provide categorical covariates keys once again. Refer to advanced usages for more options.
    categorical_covariates=['batch'],
    n_latent=128,
    # For encoder and decoder dims, provide a list of integers.
    encoder_dims=[128, 128],
    decoder_dims=[128, 128],
)
model
# -

# train the model
model.train(
    max_epochs=400,
    early_stopping=False,
    early_stopping_patience=20,
)

# Save the model
model.save('tmp/drvi_general_pipeline_immune_128', overwrite=True)

# # Latent space

# Load the model
model = DRVI.load('tmp/drvi_general_pipeline_immune_128', adata)

# +
embed = ad.AnnData(model.get_latent_representation(), obs=adata.obs)
sc.pp.subsample(embed, fraction=1.)  # Shuffling for better visualization of overlapping colors
             
sc.pp.neighbors(embed, n_neighbors=10, use_rep='X', n_pcs=embed.X.shape[1])
sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
sc.pp.pca(embed)
# -

sc.pl.umap(embed, color=['batch', 'final_annotation'], ncols=1, frameon=False)



# ## Chack latent dimension stats

drvi.user_utils.tl.set_latent_dimension_stats(model, embed)
embed.var.sort_values('reconstruction_effect', ascending=False)[:5]

drvi.user_utils.pl.plot_latent_dimension_stats(embed, ncols=2)



# You can check the same plot after removing vanished dimensions

drvi.user_utils.pl.plot_latent_dimension_stats(embed, ncols=2, remove_vanished=True)



# ## Plot latent dimensions

# By default, vanished dimensions are not plotted. Change arguments if you would like to.

# ### UMAP

drvi.user_utils.pl.plot_latent_dims_in_umap(embed)

# ### Heatmap

# Heatmaps can be useful to visualize general relasionship between latent dims and known categories of data

drvi.user_utils.pl.plot_latent_dims_in_heatmap(embed, 'final_annotation', title_col='title')

# It is possible to sort dimensions based on the top relevance with respect to a categoricals variable

drvi.user_utils.pl.plot_latent_dims_in_heatmap(embed, 'final_annotation', title_col='title', sort_by_categorical=True)





# # Interpretability

traverse_adata = drvi.user_utils.tl.traverse_latent(model, embed, n_samples=1, max_noise_std=0.)
drvi.user_utils.tl.calculate_differential_vars(traverse_adata)
traverse_adata

drvi.user_utils.pl.differential_vars_heatmap(traverse_adata, key='combined_score', score_threshold=1e-8, remove_unaffected=True, vmax=1)

drvi.user_utils.pl.show_top_differential_vars(traverse_adata, key='combined_score', score_threshold=0.)



# +
from gprofiler import GProfiler
gp = GProfiler(return_dataframe=True)

for dim_title, gene_scores in drvi.user_utils.general.iterate_on_top_differential_vars(traverse_adata, key='combined_score', score_threshold=0.):
    print(dim_title)

    gene_scores = gene_scores[gene_scores > gene_scores.max() / 10]
    print(gene_scores)
    
    relevant_genes = gene_scores.index.to_list()[:100]

    relevant_pathways = gp.profile(organism='hsapiens', query=relevant_genes, 
                                   background=list(adata.var.index), domain_scope='custom')
    display(relevant_pathways[:10])
# -


# Deeper look into genes and scores.

drvi.user_utils.pl.show_differential_vars_scatter_plot(
    traverse_adata, 
    key_x='max_possible', 
    key_y='min_possible', 
    key_combined='combined_score',
    dim_subset=['DR 10+', 'DR 26+', 'DR 38+'],
    score_threshold=0.)


drvi.user_utils.pl.plot_relevant_genes_on_umap(
    adata, embed, traverse_adata, traverse_adata_key='combined_score', 
    dim_subset=['DR 10+', 'DR 26+', 'DR 38+'],
    score_threshold=0.
)





