# Model

The DRVI (Disentangled Representation Variational Inference) model is a model designed for single-cell omics data analysis. It provides disentangled latent representations that separate individual biological processes, enabling better interpretation and downstream analysis.

## Overview

DRVI extends the standard variational autoencoder architecture with specialized decoder architecture. The model learns disentangled representations and separates different sources of variation in the data, such as:

- **Biological factors**: Cell types, developmental processes, perturbation responses, signaling pathways
- **Technical factors**: Background expressions, technical stress responses

## DRVI is now part of scvi-tools

As of `drvi-py` version `0.3.0`, the DRVI PyTorch model is no longer maintained in this
package. It has been contributed to [scvi-tools](https://scvi-tools.org/) and lives there as
`scvi.external.DRVI` (requires `scvi-tools >= 1.5.0`). New code should import the model directly:

```python
from scvi.external import DRVI
```

For backward compatibility, `drvi.model.DRVI` remains importable as an alias for
`scvi.external.DRVI`.
See the
[scvi-tools DRVI documentation](https://docs.scvi-tools.org/en/stable/api/reference/scvi.external.DRVI.html)
for the full model API.

Everything else in this package — the utility, plotting, metrics, and interpretability tools
documented in the rest of this API reference — continues to be maintained here and works on top of
the scvi-tools model.

```{eval-rst}
.. currentmodule:: drvi

.. autosummary::
    :nosignatures:
    :toctree: generated

    model.DRVI
```

## Usage Example

```python
import anndata as ad
from scvi.external import DRVI

# Load your data
adata = ad.read_h5ad("your_data.h5ad")

# Setup anndata
DRVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
)

# Initialize the model
model = DRVI(
    adata,
    n_latent=64,
    n_hidden=128,
    n_layers=2,
)

# Train the model
model.train(max_epochs=400)

# Get disentangled representations
latent = model.get_latent_representation()

# Please check tutorials for more details and downstream steps
```
