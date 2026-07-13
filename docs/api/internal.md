# Internal (developmental)

```{warning}
**For developmental internal use only.** Everything in {mod}`drvi.internal` layers experimental,
unstable extensions on top of the maintained {class}`scvi.external.DRVI` model. Its API may change
or be removed without notice. For general use, import the model as `scvi.external.DRVI` (or its
alias `drvi.model.DRVI`) and use the utilities in {mod}`drvi.utils`.
```

## Overview

`drvi.internal.DRVI` subclasses `scvi.external.DRVI` and re-adds a few features kept from earlier
drvi-py, inheriting everything else unchanged:

- **Residual connections** (`residual=True`) — since `n_hidden` is fixed across hidden layers, skip
  connections are added between the same-width hidden layers of the encoder and decoder bodies.
  Requires `n_layers >= 2` to have any effect.
- **Streaming (online) metrics** (`track_streaming_metrics=True`, the default) — per-epoch latent
  statistics (non-vanished dimension counts) and, when the data was set up with a `labels_key`,
  streaming label/latent mutual-information scores, computed during training and logged to the
  model history.
- **Sparse latent representation** — `get_sparse_latent_representation` /
  `generate_sparse_latent_representation`, which threshold small latent means and return
  `scipy.sparse` matrices.
- **Gene-subsampled reconstruction** (`n_genes_to_reconstruct=N`) — each training step reconstructs
  a random subset of `N` genes (`None` = all genes), so the decoder's `n_hidden → n_genes`
  projection runs on the subset only. Useful for scalable training on very wide panels; validation
  and all inference paths stay dense.
- **Gradient scaling** (`gradient_scale`) — multiply the gradient flowing from the decoder heads
  back into the decoder body and encoder by a fixed factor (the forward pass is unchanged); `1.0`
  is a no-op.

## DRVI (internal)

```{eval-rst}
.. currentmodule:: drvi

.. autosummary::
    :nosignatures:
    :toctree: generated

    internal.DRVI
    internal.DRVIModule
    internal.DecoderDRVI
    internal.enable_residual
    internal.LatentStats
    internal.StreamingPairwiseMI
```

## Usage Example

```python
import drvi.internal

drvi.internal.DRVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="batch",
    labels_key="cell_type",  # enables streaming MI metrics
)
model = drvi.internal.DRVI(
    adata,
    n_latent=32,
    n_layers=2,
    residual=True,  # skip connections between hidden layers
)
model.train()

# streaming metrics are in the training history
model.history["non_vanished_validation"]

# sparse latent representation (useful with a non-negative mean_activation)
z_sparse = model.get_sparse_latent_representation(zero_threshold=0.1)
```
