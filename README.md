<h1 align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/theislab/DRVI/main/.github/misc/logo.svg">
        <img width="300" src="https://raw.githubusercontent.com/theislab/DRVI/main/.github/misc/logo.svg" alt="DRVI">
    </picture>
</h1>

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

# DRVI (Unsupervised Deep Disentangled Representation of Single-Cell Omics)

<h1 align="center">
    <picture>
        <source srcset="https://raw.githubusercontent.com/theislab/DRVI/main/.github/misc/concept.svg">
        <img width="800" src="https://raw.githubusercontent.com/theislab/DRVI/main/.github/misc/concept.svg" alt="DRVI concept">
    </picture>
</h1>

## Getting started

Please refer to the [documentation][]. In particular, the

- [Tutorials][], specially
    - [Train DRVI and interpret the latent dimensions](https://drvi.readthedocs.io/latest/tutorials/external/general_pipeline.html). [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/general_pipeline.ipynb)
    - [Mapping query data into a DRVI reference](https://drvi.readthedocs.io/latest/tutorials/external/query_to_reference_mapping.html). [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/query_to_reference_mapping.ipynb)
    - [Finding rare cell types with DRVI](https://drvi.readthedocs.io/latest/tutorials/external/find_rare_cell_types.html). [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/find_rare_cell_types.ipynb)
    - Identification and annotation of factors:
        - [Cell types](https://drvi.readthedocs.io/latest/tutorials/external/identification_of_factors_1_cell_types.html). [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/identification_of_factors_1_cell_types.ipynb)
        - [Biological processes](https://drvi.readthedocs.io/latest/tutorials/external/identification_of_factors_2_biological_processes.html). [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/identification_of_factors_2_biological_processes.ipynb)
        - [Using LLM tools](https://drvi.readthedocs.io/latest/tutorials/external/identification_of_factors_3_llm_tools.html). [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/identification_of_factors_3_llm_tools.ipynb)
        - [Curation of factor annotations](https://drvi.readthedocs.io/latest/tutorials/external/identification_of_factors_4_curation.html). [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/identification_of_factors_4_curation.ipynb)
- [API documentation][], specially
    - [DRVI Model](https://drvi.readthedocs.io/latest/api/generated/drvi.model.DRVI.html)
    - [DRVI utility functions (tools)](https://drvi.readthedocs.io/latest/api/tools.html)
    - [DRVI plotting functions](https://drvi.readthedocs.io/latest/api/plotting.html)

## DRVI is now part of scvi-tools

The PyTorch model of DRVI has been contributed to [scvi-tools][] and will be
maintained from there. Everything else in this package, including the utility
and plotting functions, metrics, and interpretability tools, continues to be
maintained here and works on top of the scvi-tools model.

We recommend new projects import the model directly from scvi-tools as
`scvi.external.DRVI`, and keep importing the utilities and extras from this
package (`drvi.utils`, etc.). For backward compatibility, `drvi.model.DRVI`
remains importable as of version `0.3.0` and is now an alias for
`scvi.external.DRVI`, though this alias may itself be deprecated from `0.4.0`.
Existing users can keep using `drvi-py < 0.3.0`, but we recommend upgrading to
access utilities not available in older versions.

If you want to move a model trained with `drvi-py < 0.3.0` to the scvi-tools
implementation (`scvi-tools >= 1.5.0`), the
[Porting a DRVI model (drvi-py < 0.3) to scvi.external.DRVI (scvi-tools)](https://drvi.readthedocs.io/latest/tutorials/external/porting_drvi_to_scvi_tools.html)
tutorial walks through the conversion, so you can continue your analysis without
retraining. [![Open In Colab][open-in-colab]](https://colab.research.google.com/github/theislab/DRVI_tutorials/blob/main/porting_drvi_to_scvi_tools.ipynb)

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
> **Disentangling cellular heterogeneity into interpretable biological factors through structured latent representations.**
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
[scvi-tools]: https://scvi-tools.org/
[open-in-colab]: https://colab.research.google.com/assets/colab-badge.svg
