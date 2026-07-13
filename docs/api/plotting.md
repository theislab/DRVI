# Plotting

The DRVI plotting module provides visualization tools for analyzing latent representations and interpretability. These functions help researchers understand the disentangled representations learned by the DRVI model and their biological implications.

## Overview

The plotting module is organized into several categories:

- **Latent Visualization**: Functions for exploring and visualizing latent dimensions
- **Interpretability Analysis**: Tools for understanding how latent dimensions affect gene expression
- **Utility Functions**: Additional utility functions.

## Latent Dimension Analysis

The core functions are:

```{eval-rst}
.. module:: drvi.utils.plotting
    :no-index:
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    plotting.plot_latent_dimension_stats
    plotting.plot_latent_dims_in_umap
    plotting.plot_latent_dims_in_heatmap
```

### plot_latent_dimension_stats

Analyzes and visualizes statistics of latent dimensions to understand their properties and importance.

- Plots multiple statistics (reconstruction effect, max value, mean, std) across dimension ranking
- Distinguishes between vanished and non-vanished dimensions

**Use Cases:**

- Identify which latent dimensions are most important for reconstruction
- Understand the distribution of activation values across dimensions
- Detect vanished dimensions that contribute little to the model

### plot_latent_dims_in_umap

Visualizes latent dimensions as continuous variables on UMAP embeddings to understand their spatial distribution.
For each latent dimension, one UMAP plot wil be generated.

**Use Cases:**

- Understand how latent dimensions relate to cell clustering
- Identify spatial patterns in latent dimension activation

### plot_latent_dims_in_heatmap

Creates heatmap visualizations of latent dimensions across different cell groups or conditions.

- Groups cells by categorical variables (e.g., cell types, conditions)
- Supports balanced sampling for better visualization
- Configurable ordering and filtering of dimensions

**Use Cases:**

- Compare latent dimension activation across cell types
- Identify condition-specific latent patterns

## Interpretability

The core function is:

```{eval-rst}
.. module:: drvi.utils.plotting
    :no-index:
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    plotting.plot_interpretability_scores
```

### plot_interpretability_scores

Bar plots of the top genes per latent dimension for an interpretability score. It visualizes the
gene-score DataFrame returned by the DRVI model
(`scvi.external.DRVI.get_interpretability_scores`).

- Shows the top N genes with the highest score per dimension
- Skips dimensions whose maximum gene score is below a threshold

**Use Cases:**

- Identify the most important genes for each biological process
- Compare gene effects across different latent dimensions

## Utility Functions

The core functions are:

```{eval-rst}
.. module:: drvi.utils.plotting
    :no-index:
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    plotting.make_balanced_subsample
    plotting.cmap
```

### make_balanced_subsample

Creates balanced subsamples of AnnData objects with respect to a categorical variable.

- Equal sampling from each category
- Configurable minimum sample size per category

**Use Cases:**

- Create balanced samples for heatmap visualization

## Custom Colormaps

The module provides specialized colormaps designed for biological data visualization:

- **cmap.saturated_red_blue_cmap**: Enhanced red-blue diverging colormap for differential effects
- **cmap.saturated_just_sky_cmap**: Sky-blue colormap for positive-only effects
- **cmap.saturated_sky_cmap**: Sky-blue colormap on the positive side and gray colormap for the negative side
