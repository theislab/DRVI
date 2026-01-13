# DRVI

[![Build][badge-build]][link-build]
[![Tests][badge-tests]][link-tests]
[![Codecov][badge-codecov]][link-codecov]
[![Documentation][badge-docs]][documentation]
[![Python Version][badge-pyver]][pypi]
[![PyPI Downloads][badge-pypi-downloads]][link-pepy]

[badge-build]: https://github.com/theislab/drvi/actions/workflows/build.yaml/badge.svg
[badge-tests]: https://github.com/theislab/drvi/actions/workflows/test.yaml/badge.svg
[badge-codecov]: https://codecov.io/gh/theislab/drvi/graph/badge.svg?token=zeA0Sr3NVF
[badge-docs]: https://img.shields.io/readthedocs/drvi/latest.svg?label=Read%20the%20Docs
[badge-pyver]: https://img.shields.io/pypi/pyversions/drvi-py
[badge-pypi-downloads]: https://static.pepy.tech/personalized-badge/drvi-py?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=%F0%9F%93%A6+downloads

Unsupervised Deep Disentangled Representation of Single-Cell Omics

<h1 align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/theislab/DRVI/main/.github/misc/concept.svg">
        <img width="800" src="https://raw.githubusercontent.com/theislab/DRVI/main/.github/misc/concept.svg" alt="DRVI concept">
    </picture>
</h1>

## Getting started

Please refer to the [documentation][]. In particular, the

- [Tutorials][], specially
    - [A demo](https://drvi.readthedocs.io/latest/tutorials/external/general_pipeline.html) of how to train DRVI and interpret the latent dimensions.
- [Api documentation][], specially
    - [DRVI Model](https://drvi.readthedocs.io/latest/api/generated/drvi.model.DRVI.html)
    - [DRVI utility functions (tools)](https://drvi.readthedocs.io/latest/api/tools.html)
    - [DRVI plotting functions](https://drvi.readthedocs.io/latest/api/plotting.html)

## System requirements

We recommend running DRVI on a recent Linux distribution.
DRVI is actively tested on the latest LTS version of Ubuntu (currently 24.04 LTS).

<!-- TODO: remove ubuntu version later -->

For optimal performance, we highly recommend using a GPU with CUDA capabilities.
While CPU-based systems are supported, GPU-powered systems are strongly recommended for optimal performance.

## Installation

You need to have Python (versions 3.10 to 3.14 supported) installed on your system. If you don't have
Python installed, we recommend installing [uv][].

There are several alternative options to install drvi:

<!-- TODO: remove install time! -->

1. Install the latest release of `drvi-py` from [PyPI][], which should take around two minutes:

```bash
pip install drvi-py
```

1. Install the latest development version:

```bash
pip install git+https://github.com/theislab/drvi.git@main
```

Please be sure to install a version of [PyTorch][pytorch-home] that is compatible with your GPU.
Dependencies are installed automatically, please take a look at the versions for different dependencies in `pyproject.toml` if needed.

[pytorch-home]: https://pytorch.org/

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

If DRVI is helpful in your research, please consider citing the following paper:

> Moinfar, A. A. & Theis, F. J.
> **Unsupervised deep disentangled representation of single-cell omics.**
> bioRxiv 2024.11.06.622266 (2024) [doi:10.1101/2024.11.06.622266](https://doi.org/10.1101/2024.11.06.622266).

## Reproducibility

Code, notebooks, and instructions to reproduce the results from the paper are available at the [reproducibility repository][].

[link-build]: https://github.com/theislab/drvi/actions/workflows/build.yaml
[link-tests]: https://github.com/theislab/drvi/actions/workflows/test.yaml
[link-codecov]: https://codecov.io/gh/theislab/drvi
[link-pepy]: https://pepy.tech/projects/drvi-py

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/theislab/drvi/issues
[documentation]: https://drvi.readthedocs.io
[changelog]: https://drvi.readthedocs.io/latest/changelog.html
[api documentation]: https://drvi.readthedocs.io
[pypi]: https://pypi.org/project/drvi-py
[tutorials]: https://drvi.readthedocs.io/latest/tutorials/index.html
[reproducibility repository]: https://github.com/theislab/drvi_reproducibility
