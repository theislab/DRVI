# Porting test fixtures

Real `drvi-py < 0.3.0` DRVI checkpoints used by `tests/utils/test_porting.py` to exercise
`drvi.utils.port_to_scvi_tools`. Each subdirectory is a saved model directory containing a
`model.pt` in the **old** (pre-0.3.0) drvi-py format — the format the porter consumes.

| fixture | model configuration |
| --- | --- |
| `default` | defaults (gene dispersion, `decoder_reuse_weights="everywhere"`) |
| `reuse_nowhere` | `decoder_reuse_weights="nowhere"` |
| `dispersion_gene_batch` | `dispersion="gene-batch"` |
| `dispersion_gene_cell` | `dispersion="gene-cell"` |

The in-package DRVI model was removed in 0.3.0, so these cannot be regenerated from the current
tree. To regenerate them, check out a `drvi-py <= 0.2.7` tag (which still ships the model), then for
each configuration train a tiny model and `model.save(...)`:

```python
import anndata as ad, numpy as np, pandas as pd
from drvi.model import DRVI

rng = np.random.default_rng(0)
X = rng.poisson(1.0, size=(120, 20)).astype(np.float32)
adata = ad.AnnData(
    X=X,
    obs=pd.DataFrame(
        {"batch": [f"b{i % 2}" for i in range(120)], "cov": [f"c{i % 3}" for i in range(120)]},
        index=[f"cell_{i}" for i in range(120)],
    ),
    var=pd.DataFrame(index=[f"gene_{i}" for i in range(20)]),
)
adata.layers["counts"] = X.copy()
DRVI.setup_anndata(adata, layer="counts", batch_key="batch", categorical_covariate_keys=["cov"])
model = DRVI(adata, n_latent=8, encoder_dims=[32], decoder_dims=[32])  # + config kwargs per fixture
model.train(max_epochs=2, accelerator="cpu", batch_size=128, train_size=0.9)
model.save("tests/data/porting/<fixture>", overwrite=True, save_anndata=False)
```
