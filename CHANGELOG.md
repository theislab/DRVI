# Changelog

## [Unreleased]

## [0.1.8] - 2025-06-22

- Discretize latent dimension values in MI for benchmarking due to [this bug](https://github.com/scikit-learn/scikit-learn/issues/30772).
- Fix a minor issue with `drvi.utils.tl.traverse_latent`

## [0.1.7] - 2025-05-30

- Fix categorical lookup for reconstruction

## [0.1.6] - 2025-05-22

- Allow kwargs to pass through in plot_relevant_genes_on_umap
- Update project CI structure
- Extract tutorial notebooks to another repo to keep this repo clean

## [0.1.5] - 2025-05-09

- Refactor benchmarking code for better reusability
- Revert callable for mean and var activation

## [0.1.4] - 2025-04-17

- Limit anndata version for compatibility with old scvi-tools

## [0.1.3] - 2025-02-12

- Introduce mean activation to make non-negative latents possible (docs will come later)
- Better communication when Merlin is not installed
- Raise error when interpretability is called on model with continues covariates

## [0.1.2] - 2024-11-11

- No change in DRVI code
- Fix github workflow, tests, docs, and pypi publishing pipelines

## [0.1.0] - 2024-08-21

- Moved all files from repo to scverse cookiecutter project template
