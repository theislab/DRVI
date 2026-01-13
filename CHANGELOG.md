# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

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
