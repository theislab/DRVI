import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import drvi.internal
from drvi.internal import DecoderDRVI


def _make_adata(n=120, g=20, n_ct=3, n_batch=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "batch": [f"b{i % n_batch}" for i in range(n)],
                "cell_type": [f"ct{i % n_ct}" for i in range(n)],
            },
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(g)]),
    )
    adata.layers["counts"] = X.copy()
    return adata


def _train(model_kwargs, *, labels=True, max_epochs=3):
    adata = _make_adata()
    drvi.internal.DRVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="batch",
        labels_key="cell_type" if labels else None,
    )
    model = drvi.internal.DRVI(adata, n_latent=8, n_hidden=16, **model_kwargs)
    model.train(
        max_epochs=max_epochs,
        accelerator="cpu",
        batch_size=64,
        train_size=0.8,
        check_val_every_n_epoch=1,
    )
    return adata, model


class TestResidual:
    def test_residual_enabled_on_layers(self):
        _, model = _train({"n_layers": 2, "residual": True}, labels=False)
        assert getattr(model.module.z_encoder.encoder, "is_residual", False)
        assert getattr(model.module.decoder.px_decoder, "is_residual", False)
        assert model.module.residual is True

    def test_no_residual_by_default(self):
        _, model = _train({"n_layers": 2}, labels=False)
        assert not getattr(model.module.z_encoder.encoder, "is_residual", False)
        assert model.module.residual is False

    def test_residual_produces_finite_latent(self):
        adata, model = _train({"n_layers": 2, "residual": True}, labels=False)
        z = model.get_latent_representation()
        assert z.shape == (adata.n_obs, 8)
        assert np.isfinite(z).all()


class TestStreamingMetrics:
    def test_metrics_logged_with_labels(self):
        _, model = _train({"n_layers": 1}, labels=True)
        assert model.module.latent_stats is not None
        assert model.module.mi_metric is not None
        keys = set(model.history)
        # latent stats (both splits) and MI summary are logged per epoch
        assert "non_vanished_train" in keys
        assert "non_vanished_validation" in keys
        assert "lms_smi_validation" in keys

    def test_latent_stats_without_labels(self):
        _, model = _train({"n_layers": 1}, labels=False)
        assert model.module.latent_stats is not None
        assert model.module.mi_metric is None  # MI needs n_labels > 1
        assert "non_vanished_train" in set(model.history)

    def test_can_disable_metrics(self):
        _, model = _train({"n_layers": 1, "track_streaming_metrics": False}, labels=True)
        assert model.module.latent_stats is None
        assert model.module.mi_metric is None


class TestSparseLatent:
    def test_dense_sparse_matches_latent(self):
        adata, model = _train({"n_layers": 1, "mean_activation": "ReLU"}, labels=False)
        z_sparse = model.get_sparse_latent_representation()
        assert sparse.issparse(z_sparse)
        assert z_sparse.shape == (adata.n_obs, 8)

    def test_threshold_increases_sparsity(self):
        adata, model = _train({"n_layers": 1, "mean_activation": "ReLU"}, labels=False)
        dense = model.get_sparse_latent_representation(zero_threshold=0.0)
        thresholded = model.get_sparse_latent_representation(zero_threshold=1e6)
        assert thresholded.nnz <= dense.nnz
        assert thresholded.nnz == 0  # everything below 1e6 is zeroed

    def test_return_dist(self):
        adata, model = _train({"n_layers": 1}, labels=False)
        mean, var = model.get_sparse_latent_representation(return_dist=True)
        assert mean.shape == var.shape == (adata.n_obs, 8)


class TestGeneSubsampledReconstruction:
    def test_dense_by_default(self):
        _, model = _train({"n_layers": 1}, labels=False)
        assert model.module.n_genes_to_reconstruct is None
        assert not isinstance(model.module.decoder, DecoderDRVI)

    def test_subsample_trains_and_swaps_decoder(self):
        # 20 genes total, reconstruct 8 per step
        adata, model = _train({"n_layers": 1, "n_genes_to_reconstruct": 8}, labels=False)
        assert model.module.n_genes_to_reconstruct == 8
        assert isinstance(model.module.decoder, DecoderDRVI)
        assert model.history["reconstruction_loss_train"].notna().all().all()

    @pytest.mark.parametrize("dispersion", ["gene", "gene-batch", "gene-cell"])
    def test_subsample_all_dispersions(self, dispersion):
        adata, model = _train(
            {"n_layers": 1, "n_genes_to_reconstruct": 8, "dispersion": dispersion},
            labels=False,
        )
        # inference/latent path stays dense and finite
        z = model.get_latent_representation()
        assert z.shape == (adata.n_obs, 8)
        assert np.isfinite(z).all()

    def test_inference_paths_stay_dense(self):
        # decode over all genes even though training subsampled
        adata, model = _train({"n_layers": 1, "n_genes_to_reconstruct": 8}, labels=False)
        model.module.eval()  # not training -> dense
        assert model.module._get_reconstruction_indices({}) is None
        norm = model.get_normalized_expression()
        assert norm.shape == (adata.n_obs, adata.n_vars)


class TestGradientScale:
    def test_no_hook_by_default(self):
        _, model = _train({"n_layers": 1}, labels=False)
        assert model.module.gradient_scale == 1.0
        assert len(model.module.decoder.px_decoder._forward_hooks) == 0

    def test_hook_registered_and_trains(self):
        adata, model = _train({"n_layers": 1, "gradient_scale": 0.5}, labels=False)
        assert model.module.gradient_scale == 0.5
        assert len(model.module.decoder.px_decoder._forward_hooks) == 1
        z = model.get_latent_representation()
        assert z.shape == (adata.n_obs, 8)
        assert np.isfinite(z).all()

    def test_hook_is_identity_forward_scaled_backward(self):
        import torch

        _, model = _train({"n_layers": 1, "gradient_scale": 0.5}, labels=False)
        hook = next(iter(model.module.decoder.px_decoder._forward_hooks.values()))
        x = torch.randn(3, 4, requires_grad=True)
        out = hook(None, None, x)
        assert torch.allclose(out, x)  # forward value unchanged
        out.sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, 0.5))  # gradient scaled by 0.5


@pytest.mark.parametrize("residual", [False, True])
def test_end_to_end_smoke(residual):
    adata, model = _train({"n_layers": 2, "residual": residual}, labels=True)
    z = model.get_latent_representation()
    assert z.shape == (adata.n_obs, 8)


def test_save_load_roundtrip(tmp_path):
    adata, model = _train({"n_layers": 2, "residual": True}, labels=True)
    z_before = model.get_latent_representation()
    model.save(str(tmp_path / "m"), overwrite=True)

    reloaded = drvi.internal.DRVI.load(str(tmp_path / "m"), adata=adata)
    # architecture flags survive the round-trip (init_params_ keyed on the subclass signature)
    assert reloaded.module.n_latent == 8
    assert reloaded.module.residual is True
    assert getattr(reloaded.module.z_encoder.encoder, "is_residual", False)
    assert reloaded.module.latent_stats is not None and reloaded.module.mi_metric is not None
    assert np.allclose(z_before, reloaded.get_latent_representation(), atol=1e-5)
