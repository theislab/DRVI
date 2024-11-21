# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: drvi-repr
#     language: python
#     name: drvi-repr
# ---

# # General training and interpretability pipeline

# In this notebook, we analyze the immune dataset of 9 batches from four human peripheral blood and bone marrow studies, with 16 annotated cell types. We apply DRVI with 128 latent dimensions to showcase the following:
#
# - How to train DRVI
# - Observe vanished dimensions
# - Observe the latent space in UMAP and heatmap
# - How to run the interpretability pipeline
# - How to identify and check individual dimensions

# ## Imports

# %load_ext autoreload
# %autoreload 2

import warnings

warnings.filterwarnings("ignore")

# +
import anndata as ad
import scanpy as sc
from matplotlib import pyplot as plt
from IPython.display import display
from gprofiler import GProfiler

import drvi
from drvi.model import DRVI
from drvi.utils.misc import hvg_batch
# -

sc.settings.set_figure_params(dpi=100, frameon=False)
sc.set_figure_params(dpi=100)
sc.set_figure_params(figsize=(3, 3))
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.figsize"] = (3, 3)

# ## Load Data

# + language="bash"
#
# # Download Immune dataset
#
# mkdir -p tmp
#
# # Check if the file exists
# if [ ! -f tmp/immune_all.h5ad ]; then
#   # Download the file if it does not exist
#   { # try
#       wget -O tmp/immune_all.h5ad https://figshare.com/ndownloader/files/25717328
#       #save your output
#   } || \
#   { # catch
#       curl -L https://figshare.com/ndownloader/files/25717328 -o tmp/immune_all.h5ad
#   }
#   echo "File downloaded successfully."
# else
#   echo "File already exists."
# fi
# -

adata = sc.read("tmp/immune_all.h5ad")
# Remove dataset with non-count values
adata = adata[adata.obs["batch"] != "Villani"].copy()
adata

# ## Pre-processing

adata.X = adata.layers["counts"].copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata

sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
adata

# Batch aware HVG selection (method is obtained from scIB metrics)
hvg_genes = hvg_batch(adata, batch_key="batch", target_genes=2000, adataOut=False)
adata = adata[:, hvg_genes].copy()
adata


sc.pl.umap(adata, color=["batch", "final_annotation"], ncols=1, frameon=False)


# ## Train DRVI

# +
# Setup data
DRVI.setup_anndata(
    adata,
    # DRVI accepts count data by default.
    # Do not forget to change gene_likelihood if you provide a non-count data.
    layer="counts",
    # Always provide a list. DRVI can accept multiple covariates.
    categorical_covariate_keys=["batch"],
    # DRVI accepts count data by default.
    # Set to false if you provide log-normalized data and use normal distribution (mse loss).
    is_count_data=True,
)

# construct the model
model = DRVI(
    adata,
    # Provide categorical covariates keys once again. Refer to advanced usages for more options.
    categorical_covariates=["batch"],
    n_latent=128,
    # For encoder and decoder dims, provide a list of integers.
    encoder_dims=[128, 128],
    decoder_dims=[128, 128],
)
model

# +
# For cpu training you should add the following line to the model.train parameters:
# accelerator="cpu", devices=1,
#
# For mps acceleration on macbooks, add the following line to the model.train parameters:
# accelerator="mps", devices=1,
#
# For gpu training don't provide any additional parameter.
# More details here: https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html

n_epochs = 400

# train the model
model.train(
    max_epochs=n_epochs,
    early_stopping=False,
    early_stopping_patience=20,
    # mps
    # accelerator="mps", devices=1,
    # cpu
    # accelerator="cpu", devices=1,
    # gpu: no additional parameter
    #
    # No need to provide `plan_kwargs` if n_epochs >= 400.
    plan_kwargs={
        "n_epochs_kl_warmup": n_epochs,
    },
)

# Runtime:
# The runtime for CPU laptop (M1) is 208 minutes
# The runtime for Macbook gpu (M1) is 64 minutes
# The runtime for GPU (A100) is 17 minutes
# -

# Save the model
model.save("tmp/drvi_general_pipeline_immune_128", overwrite=True)

# ## Latent space

# Load the model
model = DRVI.load("tmp/drvi_general_pipeline_immune_128", adata)

# +
embed = ad.AnnData(model.get_latent_representation(), obs=adata.obs)
sc.pp.subsample(embed, fraction=1.0)  # Shuffling for better visualization of overlapping colors

sc.pp.neighbors(embed, n_neighbors=10, use_rep="X", n_pcs=embed.X.shape[1])
sc.tl.umap(embed, spread=1.0, min_dist=0.5, random_state=123)
sc.pp.pca(embed)

embed.write("tmp/drvi_general_pipeline_immune_128_embed.h5ad")
# -

embed = sc.read("tmp/drvi_general_pipeline_immune_128_embed.h5ad")

sc.pl.umap(embed, color=["batch", "final_annotation"], ncols=1, frameon=False)


# ### Chack latent dimension stats

drvi.utils.tl.set_latent_dimension_stats(model, embed)
embed.var.sort_values("reconstruction_effect", ascending=False)[:5]

drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2)


# You can check the same plot after removing vanished dimensions

drvi.utils.pl.plot_latent_dimension_stats(embed, ncols=2, remove_vanished=True)


# ### Plot latent dimensions

# By default, vanished dimensions are not plotted. Change arguments if you would like to.

# #### UMAP

drvi.utils.pl.plot_latent_dims_in_umap(embed)

# #### Heatmap

# Heatmaps can be useful to visualize general relationship between latent dims and known categories of data

drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "final_annotation", title_col="title")

# It is possible to sort dimensions based on the top relevance with respect to a categoricals variable

drvi.utils.pl.plot_latent_dims_in_heatmap(embed, "final_annotation", title_col="title", sort_by_categorical=True)


# ## Interpretability

# ### Traversing the latent space

# Here we use DRVI's utils to traverse latent space and find the effect of each latent dimension

traverse_adata = drvi.utils.tl.traverse_latent(model, embed, n_samples=20, max_noise_std=0.0)
drvi.utils.tl.calculate_differential_vars(traverse_adata)
traverse_adata.write("tmp/drvi_general_pipeline_immune_128_traverse_adata.h5ad")

traverse_adata = sc.read("tmp/drvi_general_pipeline_immune_128_traverse_adata.h5ad")
traverse_adata

# ### Getting the results

# We can then visualize the top relevant genes for each dimension

drvi.utils.pl.show_top_differential_vars(traverse_adata, key="combined_score", score_threshold=0.0)


# ### Identify using external tools

# Additionally any gene set enrichment tool / marker gene searching tools can be used to identify the meaning of dimensions

# +
gp = GProfiler(return_dataframe=True)
dimensions_interpretability = drvi.utils.tools.iterate_on_top_differential_vars(
    traverse_adata, key="combined_score", score_threshold=0.0
)

# For making it brief we just iterate over 5 dimensions
for dim_title, gene_scores in dimensions_interpretability[:5]:
    print(dim_title)

    gene_scores = gene_scores[gene_scores > gene_scores.max() / 10]
    print(gene_scores)

    relevant_genes = gene_scores.index.to_list()[:100]

    relevant_pathways = gp.profile(
        organism="hsapiens", query=relevant_genes, background=list(adata.var.index), domain_scope="custom"
    )
    display(relevant_pathways[:10])
# -


# ### Looking into individual dimensions

# A user can take a deeper look into individual dimensions. we can see the min_possible, and max_possible log-fold-changes of each dimension. In addition, one can check the activity of top relevant genes for dimensions of interest.

# We visualize 3 dimensions:
# 1. DR 12+ highlighting CD8
# 2. DR 22- highlighting Dissociation stress response
# 3. DR 23+ highlighting T-reg cells
drvi.utils.pl.show_differential_vars_scatter_plot(
    traverse_adata,
    key_x="max_possible",
    key_y="min_possible",
    key_combined="combined_score",
    dim_subset=["DR 12+", "DR 22-", "DR 23+"],
    score_threshold=0.0,
)


drvi.utils.pl.plot_relevant_genes_on_umap(
    adata,
    embed,
    traverse_adata,
    traverse_adata_key="combined_score",
    dim_subset=["DR 12+", "DR 22-", "DR 23+"],
    score_threshold=0.0,
)
