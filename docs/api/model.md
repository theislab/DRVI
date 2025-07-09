# Model

The DRVI (Disentangled Representation Variational Inference) model is a model designed for single-cell omics data analysis. It provides disentangled latent representations that separate individual biological processes, enabling better interpretation and downstream analysis.

## Overview

DRVI extends the standard variational autoencoder architecture with specialized decoder architecture. The model learns disentangled representations and separates different sources of variation in the data, such as:

- **Biological factors**: Cell types, developmental processes, perturbation responses, signaling pathways
- **Technical factors**: Background expressions, technical stress responses

## Core Components

### DRVI model

This is the main model class that can be used to define, train, and evaluate the model on an anndata. `DRVI` passes any extra argument to `DRVIModule` in initialization. Accordingly, we suggest to check its documentation (below) for additional configurations.

```{eval-rst}
.. module:: drvi.model
    :no-index:
.. currentmodule:: drvi

.. autosummary::
    :nosignatures:
    :toctree: generated

    model.DRVI
```

### DRVIModule

This is the pytorch neural network module and contains DRVI logic.

```{eval-rst}
.. module:: drvi.model
.. currentmodule:: drvi

.. autosummary::
    :nosignatures:
    :toctree: generated

    model.DRVIModule
```

## Usage Example

```python
import anndata as ad
from drvi.model import DRVI

# Load your data
adata = ad.read_h5ad("your_data.h5ad")

# Setup anndata
DRVI.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=["batch"],
    is_count_data=True,
)

# Initialize the model
model = DRVI(
    adata,
    categorical_covariates=["batch"],
    n_latent=64,
    encoder_dims=[128, 128],
    decoder_dims=[128, 128],
)

# Train the model
model.train(
    max_epochs=400,
    early_stopping=False,
)

# Get disentangled representations
latent = model.get_latent_representation()

# Please check tutorials for more details and downstream steps
```
