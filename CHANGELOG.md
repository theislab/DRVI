# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added


## [0.3.0]

### Changed
- The DRVI PyTorch model is no longer shipped in this package. It has been contributed to [scvi-tools](https://scvi-tools.org/) and now lives there as `scvi.external.DRVI` (requires `scvi-tools >= 1.5.0`, now the minimum dependency). `drvi.model.DRVI` is kept as an alias for `scvi.external.DRVI` for backward compatibility and may be deprecated from `0.4.0`. New code should import the model directly as `scvi.external.DRVI`. Utility, plotting, metrics, and interpretability tools (`drvi.utils`) continue to be maintained here and work on top of the scvi-tools model.

### Removed
- Removed the in-package model implementation and its internals: `drvi.scvi_tools_based` (model, module, training plan, neural-network components, data fields), `drvi.nn_modules`, and `drvi.model.DRVIModule`. Model capabilities not supported by `scvi.external.DRVI` are dropped and will not be reinstated. To move a model trained with `drvi-py < 0.3.0` to scvi-tools, use `drvi.utils.port_to_scvi_tools`.
- Emptied the `drvi.utils.tl` tools namespace (`drvi.utils.tools`): removed `set_latent_dimension_stats`, `traverse_latent`, `calculate_differential_vars`, `get_split_effects`, and `iterate_on_top_differential_vars`. The latent-traversal interpretability pipeline is superseded by the scvi-tools DRVI model's built-in `set_latent_dimension_stats` / `calculate_interpretability_scores` / `get_interpretability_scores`. The `drvi.utils.tl` namespace is kept (now empty) as a stable import location.
- Removed the traversal-based plotting functions `drvi.utils.pl.show_top_differential_vars`, `plot_relevant_genes_on_umap`, `show_differential_vars_scatter_plot`, `differential_vars_heatmap`, and `make_heatmap_groups` (they consumed the removed traversal outputs). `plot_interpretability_scores` (which visualizes the model's `get_interpretability_scores`) and the latent-dimension plots remain.


## [0.2.7]

### Added
- Add `drvi.utils.port_to_scvi_tools` to migrate a trained/saved drvi-py model into a checkpoint loadable by `scvi.external.DRVI` (available in scvi-tools 1.5.0). Pure checkpoint surgery — no model is instantiated. Unsupported capabilities raise `DRVIPortError` with a link to open an issue.


## [0.2.6] - 2026-07-02

### Added
- Add `drvi.utils.pl.plot_interpretability_scores` plotting utility, now reused by `DRVI.plot_interpretability_scores`
- Add `binary_maximum_mutual_information_score` (BMMI) disentanglement metric
- Add `min_max_thresholds` argument to `plot_latent_dims_in_umap` to control color-scale clamping
- Add `"gelu"` option and callable factory support for the encoder `mean_activation`

### Changed
- In `DiscreteDisentanglementBenchmark`, default metrics are now `("SMI", "SPN")`, samples are shuffled before evaluation, and the benchmark version is bumped to `v3_1`
- Renamed `discrete_mutual_info_score` to `discrete_scaled_mutual_info_score`.
- Unknown encoder `mean_activation` values now raise `NotImplementedError` instead of failing an assertion
- `LatentStats` now logs `non_vanished*` counts as Python scalars

### Removed
- Remove MI metrics `local_mutual_info_score`, and `global_dim_mutual_info_score`, along with the `SMI-cont`/`SMI-disc` benchmark keys (superseded by `SMI` and `BMMI`)

### Fixed
- Fix typo in metric name `spearman_correlataion_score` -> `spearman_correlation_score`


## [0.2.5] - 2026-04-21

### Added
- latent stat (n_vanished) logs during training
- Disentanglement metric logs during training
- Relax upper bounds of all dependencies

## [0.2.4] - 2026-03-02

### Added
- Allow sparse input as X
- Allow generating memory efficient sparse tensors as latent

## [0.2.3] - 2026-02-27

### Added
- Added within distribution interpretability
- An efficient implementation of out-of-distribution interpretability is added
- plotting and getting relevant genes is now possible in DRVI model interface.
- Setting latent dimension stats is now done in model interface. Previous util functions still work, but will show a deprecation warning.
- Add tutorial for query to reference mapping

### Changed
- The default value for vanished threshold in new the interface of `set_latent_dimension_stats` is 0.5 (previously 0.1).
- Interpretability scores for the other direction of a non-vanished dimension is not shown if that direction is meaningless.
- Main tutorial updated with the new interpretability interface

## [0.2.2] - 2026-02-11

### Added
- Added "tutorials" optional dependency that was accidentally removed.
- Code to allow loading models from previous versions with no problem
- Add dispersion parameter to allow modeling batch dependent dispersion

### Removed
- Removed "x_loglib", "div_lib_x_loglib", "x_loglib_all" library normalization techniques
- Removed Vamp prior and GMM prior
- Removed Legacy likelihood functions. Will raise error if used.

### Changed
- gene_likelihood parameter now accepts different values compared to before. gene_likelihood parameter from old models will be mapped properly, but new users should look into the docs.

## [0.2.1] - 2025-12-29

### Added

- Add support for Python 3.13
- Allow subset reconstruction
- Allow gradient scaling in the last layer
- Allow setting vector size after mapping in "split_map@k" and "power@k" splitting functions.
- Add support for Python 3.14

### Removed
- Remove restrict dependencies. To ensure compatibility with old packages run for example: `uvx --exclude-newer 2024-01-01 hatch run pytest`

### Changed

- Minor code improvements
- Update to scverse template version 0.7.0

## [0.2.0] - 2025-11-24

### Changed

- Update upper bound of all dependencies
- Align with the latest changes of scvi-tools

### Added

- Use cruft for scverse template management

## [0.1.11] - 2025-11-19

### Added

- Add RnaSeqMixin from scvi-tools for RNA-seq specific methods (get_normalized_expression, differential_expression, posterior_predictive_sample, get_likelihood_parameters)

### Fixed

- Fix bug in decode space handling where library size was not considered (issue #46)
- Update decode_latent_samples logic (decode in log space by default)
- Code improvements and bug fixes

## [0.1.10] - 2025-11-18

### Removed
- Remove merlin data support and all related code
- Remove merlin-dataloader dependency

### Added

- Add dependabot for dependency update notification (not used now, for next releases)

## [0.1.9] - 2025-07-01

### Added

- Add DRVI-APnoEXP baseline

### Fixed

- Imorove documnetation for all classes and functions in repository

## [0.1.8] - 2025-06-22

### Changed

- Discretize latent dimension values in MI for benchmarking due to [this bug](https://github.com/scikit-learn/scikit-learn/issues/30772).

### Fixed

- Fix a minor issue with `drvi.utils.tl.traverse_latent`

## [0.1.7] - 2025-05-30

### Fixed

- Fix categorical lookup for reconstruction

## [0.1.6] - 2025-05-22

### Changed

- Allow kwargs to pass through in plot_relevant_genes_on_umap
- Update project CI structure
- Extract tutorial notebooks to another repo to keep this repo clean

## [0.1.5] - 2025-05-09

### Changed

- Refactor benchmarking code for better reusability
- Revert callable for mean and var activation

## [0.1.4] - 2025-04-17

### Fixed

- Limit anndata version for compatibility with old scvi-tools

## [0.1.3] - 2025-02-12

### Added

- Introduce mean activation to make non-negative latents possible (docs will come later)

### Fixed

- Better communication when Merlin is not installed
- Raise error when interpretability is called on model with continues covariates

## [0.1.2] - 2024-11-11

### Fixed

- No change in DRVI code
- Fix github workflow, tests, docs, and pypi publishing pipelines

## [0.1.0] - 2024-08-21

### Added

- Moved all files from repo to scverse cookiecutter project template
