# Tools

The DRVI tools module provides utilities for analyzing and interpreting latent representations.

## Overview

The tools module is organized into two main categories:

- **Latent Dimension Analysis**: Functions for analyzing and characterizing latent dimensions
- **Interpretability Tools**: Functions for understanding how latent dimensions affect gene expression

## Latent Dimension Analysis

```{eval-rst}
.. module:: drvi.utils.tools
    :no-index:
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    tools.set_latent_dimension_stats
```

### set_latent_dimension_stats

Analyzes and characterizes latent dimensions by computing various statistics.

- Calculates basic statistics and reconstruction effect for each dimension
- Identifies vanished dimensions that contribute little to the model
- Provides ranking and ordering of dimensions by importance
- Essential for understanding

**Use Cases:**

- Identify and filter out non-informative dimensions
- Rank dimensions for downstream analysis and visualization

## Interpretability Tools

```{eval-rst}
.. module:: drvi.utils.tools
    :no-index:
.. currentmodule:: drvi.utils

.. autosummary::
    :nosignatures:
    :toctree: generated

    tools.traverse_latent
    tools.calculate_differential_vars
    tools.get_split_effects
    tools.iterate_on_top_differential_vars
```

### traverse_latent

Performs systematic traversals through latent dimensions to understand their effects on gene expression.

- Systematically varies each latent dimension while keeping others fixed
- Generates synthetic data points across the latent space
- Enables analysis of how individual dimensions affect gene expression
- Should be used with the next function

**Use Cases:**

- Understand how each latent dimension affects gene expression

### calculate_differential_vars

Identifies genes that are differentially affected by latent dimension changes (traverses).

- Computes various differential effect metrics (max_possible, min_possible, combined_score)
- Identifies genes most relevant to each latent dimension

**Use Cases:**

- Identify genes related to specific biological processes
- Quantify the strength of gene-latent dimension relationships
- Generate gene lists for downstream biological analysis

### get_split_effects

This function is simply the combination of `traverse_latent` and `calculate_differential_vars`.

### iterate_on_top_differential_vars

Iterative over top relevant genes.

**Use Cases:**

- Can be used to construct a for loop over top relevant dimensions.
- It can be used along other tools for biological interpretations of latent dimensions
