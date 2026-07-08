"""Tests for :func:`drvi.utils.port_to_scvi_tools` (drvi-py -> scvi-tools checkpoint migration).

These validate the checkpoint transformation itself (key remapping, value transforms, ``init_params_``
and ``registry_`` rewriting) without importing scvi-tools. An end-to-end numerical-equivalence check
against ``scvi.external.DRVI`` lives outside the drvi-py test suite (it needs scvi-tools installed).
"""

import copy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from drvi.model import DRVI
from drvi.utils import port_to_scvi_tools
from drvi.utils.porting import DRVIPorter, DRVIPortError


def _make_adata(n=200, g=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.poisson(1.0, size=(n, g)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "batch": [f"b{i % 2}" for i in range(n)],
                "cov": [f"c{i % 3}" for i in range(n)],
            },
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(g)]),
    )
    adata.layers["counts"] = X.copy()
    return adata


def _train_and_save(tmp_path, name, **model_kwargs):
    adata = _make_adata()
    DRVI.setup_anndata(adata, layer="counts", batch_key="batch", categorical_covariate_keys=["cov"])
    model = DRVI(adata, **model_kwargs)
    model.train(max_epochs=2, accelerator="cpu", batch_size=128, train_size=0.9)
    out = str(tmp_path / name)
    model.save(out, overwrite=True)
    return out


def _load_sd(model_dir):
    ckpt = torch.load(f"{model_dir}/model.pt", map_location="cpu", weights_only=False)
    return ckpt["model_state_dict"], ckpt["attr_dict"], list(ckpt["var_names"])


class TestPortToScviTools:
    def test_default_model_end_to_end(self, tmp_path):
        src = _train_and_save(tmp_path, "src_default")
        old_sd, _, var_names = _load_sd(src)

        dest = port_to_scvi_tools(src)  # default dest = "{src}_scvi_tools"
        assert dest == f"{src}_scvi_tools"
        new_sd, new_attr, new_var = _load_sd(dest)

        # var_names preserved
        assert new_var == var_names

        # --- key remapping: old-only prefixes gone, new names present ---
        for banned in (
            "decoder.params_nets",
            "decoder.px_shared_decoder",
            "decoder.split_transformation_weight",
            "z_encoder.mean_encoder.fc_layers",
            "z_encoder.var_encoder.fc_layers",
            "latent_stats",
            "mi_metric",
            "prior.",
        ):
            assert not any(k.startswith(banned) for k in new_sd), banned

        # --- encoder body: identity ---
        for k, v in old_sd.items():
            if k.startswith("z_encoder.encoder.fc_layers."):
                assert torch.equal(new_sd[k], v)

        # --- mean/var encoder un-nested (FCLayers -> plain Linear) ---
        assert torch.equal(
            new_sd["z_encoder.mean_encoder.weight"],
            old_sd["z_encoder.mean_encoder.fc_layers.Layer 0.0.weight"],
        )
        assert torch.equal(
            new_sd["z_encoder.var_encoder.bias"],
            old_sd["z_encoder.var_encoder.fc_layers.Layer 0.0.bias"],
        )

        # --- split transform: transposed (n_split, in, out) -> (n_split, out, in) ---
        assert torch.equal(
            new_sd["decoder.split_transform.weight"],
            old_sd["decoder.split_transformation_weight"].transpose(-1, -2),
        )

        # --- scale head un-nested ---
        assert torch.equal(
            new_sd["decoder.px_scale_decoder.weight"],
            old_sd["decoder.params_nets.mean.fc_layers.Layer 0.0.weight"],
        )

        # --- dispersion "gene": px_r = old r + 1.0 (drvi's r = 1.0 + param; scvi uses px_r directly) ---
        assert torch.allclose(new_sd["px_r"], old_sd["decoder.params_nets.r"] + 1.0)

        # --- synthesized (unused) library encoder present and zeroed ---
        for k in (
            "l_encoder.encoder.fc_layers.Layer 0.0.weight",
            "l_encoder.encoder.fc_layers.Layer 0.0.bias",
            "l_encoder.mean_encoder.weight",
            "l_encoder.var_encoder.weight",
        ):
            assert k in new_sd
            assert torch.count_nonzero(new_sd[k]) == 0

        # --- init_params_ rewritten to scvi-tools DRVI signature ---
        non_kwargs = new_attr["init_params_"]["non_kwargs"]
        assert set(non_kwargs) == {
            "n_latent",
            "n_split_latent",
            "split_method",
            "split_aggregation",
            "gene_likelihood",
        }
        assert non_kwargs["split_method"] == "split_map"
        assert non_kwargs["split_aggregation"] == "logsumexp"
        kwargs = new_attr["init_params_"]["kwargs"]["kwargs"]
        assert kwargs["dispersion"] == "gene"
        assert kwargs["decoder_reuse_weights"] == "everywhere"
        assert kwargs["use_observed_lib_size"] is True
        assert kwargs["batch_representation"] == "one-hot"

        # --- registry cleaned for scvi-tools setup_anndata ---
        registry = new_attr["registry_"]
        assert registry["model_name"] == "DRVI"
        assert registry["setup_method_name"] == "setup_anndata"
        assert "is_count_data" not in registry["setup_args"]
        assert registry["setup_args"]["size_factor_key"] is None
        assert new_attr["is_trained_"] is True

    def test_no_weight_sharing_transposes_head(self, tmp_path):
        src = _train_and_save(tmp_path, "src_nowhere", decoder_reuse_weights="nowhere")
        old_sd, _, _ = _load_sd(src)
        dest = port_to_scvi_tools(src)
        new_sd, new_attr, _ = _load_sd(dest)

        old_head = old_sd["decoder.params_nets.mean.fc_layers.Layer 0.0.weight"]
        assert old_head.dim() == 3  # per-split StackedLinearLayer weight
        assert torch.equal(new_sd["decoder.px_scale_decoder.weight"], old_head.transpose(-1, -2))
        assert new_attr["init_params_"]["kwargs"]["kwargs"]["decoder_reuse_weights"] == "hidden"

    def test_gene_batch_dispersion_folds_offset(self, tmp_path):
        # scvi computes gene-batch dispersion as linear(one_hot(batch), px_r) with no bias/offset,
        # so the old Linear weight, bias, and drvi's +1.0 must fuse into one px_r matrix.
        src = _train_and_save(tmp_path, "src_gb", dispersion="gene-batch")
        old_sd, _, _ = _load_sd(src)
        dest = port_to_scvi_tools(src)
        new_sd, new_attr, _ = _load_sd(dest)

        w = old_sd["decoder.params_nets.r.weight"]  # (n_genes, n_batch)
        b = old_sd["decoder.params_nets.r.bias"]  # (n_genes,)
        # folded in drvi's runtime add-order (W + b) + 1 -> bit-identical to drvi's forward
        assert torch.equal(new_sd["px_r"], (w + b.unsqueeze(1)) + 1.0)
        assert not any(k.startswith("decoder.px_r_decoder") for k in new_sd)
        assert new_attr["init_params_"]["kwargs"]["kwargs"]["dispersion"] == "gene-batch"

    def test_gene_cell_dispersion_head_raw(self, tmp_path):
        # scvi adds drvi's +1.0 after split-aggregation, so the per-cell head is copied raw.
        src = _train_and_save(tmp_path, "src_gc", dispersion="gene-cell")
        old_sd, _, _ = _load_sd(src)
        dest = port_to_scvi_tools(src)
        new_sd, new_attr, _ = _load_sd(dest)

        old_w = old_sd["decoder.params_nets.r.fc_layers.Layer 0.0.weight"]
        old_b = old_sd["decoder.params_nets.r.fc_layers.Layer 0.0.bias"]
        expected_w = old_w.transpose(-1, -2) if old_w.dim() == 3 else old_w
        assert torch.equal(new_sd["decoder.px_r_decoder.weight"], expected_w)
        # +1.0 folded into the head bias (commutes through split aggregation)
        assert torch.allclose(new_sd["decoder.px_r_decoder.bias"], old_b + 1.0)
        assert "px_r" not in new_sd  # gene-cell has no module-level dispersion parameter
        assert new_attr["init_params_"]["kwargs"]["kwargs"]["dispersion"] == "gene-cell"

    def test_does_not_overwrite(self, tmp_path):
        src = _train_and_save(tmp_path, "src_ow")
        dest = str(tmp_path / "explicit_dest")
        port_to_scvi_tools(src, dest)
        with pytest.raises(FileExistsError):
            port_to_scvi_tools(src, dest)
        # overwrite=True succeeds
        port_to_scvi_tools(src, dest, overwrite=True)

    def test_raises_on_removed_capability(self, tmp_path):
        """A capability with no scvi-tools equivalent must raise DRVIPortError, not silently port."""
        src = _train_and_save(tmp_path, "src_cap")
        ckpt = torch.load(f"{src}/model.pt", map_location="cpu", weights_only=False)

        # unsupported split_aggregation
        bad = copy.deepcopy(ckpt)
        bad["attr_dict"]["init_params_"]["kwargs"]["model_kwargs"] = {"split_aggregation": "max"}
        bad_dir = tmp_path / "bad_agg"
        bad_dir.mkdir()
        torch.save(bad, bad_dir / "model.pt")
        with pytest.raises(DRVIPortError):
            port_to_scvi_tools(str(bad_dir))

        # unsupported covariate strategy
        bad2 = copy.deepcopy(ckpt)
        bad2["attr_dict"]["init_params_"]["kwargs"]["model_kwargs"] = {"covariate_modeling_strategy": "emb"}
        bad2_dir = tmp_path / "bad_cov"
        bad2_dir.mkdir()
        torch.save(bad2, bad2_dir / "model.pt")
        with pytest.raises(DRVIPortError):
            port_to_scvi_tools(str(bad2_dir))

        # decoder dropout: scvi-tools DRVI's decoder is dropout-free -> must raise
        bad3 = copy.deepcopy(ckpt)
        bad3["attr_dict"]["init_params_"]["kwargs"]["model_kwargs"] = {"decoder_dropout_rate": 0.1}
        bad3_dir = tmp_path / "bad_dec_dropout"
        bad3_dir.mkdir()
        torch.save(bad3, bad3_dir / "model.pt")
        with pytest.raises(DRVIPortError, match="decoder dropout"):
            port_to_scvi_tools(str(bad3_dir))


class TestActivationCoverage:
    """Every drvi activation (mean and FCLayers) maps to a scvi-tools value or is rejected."""

    @pytest.fixture
    def porter(self):
        return DRVIPorter.__new__(DRVIPorter)  # bare instance; the methods use only class data

    @pytest.mark.parametrize(
        ("drvi_value", "expected"),
        [
            ("identity", None),
            ("relu", "ReLU"),
            ("gelu", "GELU"),
            ("leaky_relu", "LeakyReLU"),
            ("leaky_relu_0.2", "LeakyReLU_0.2"),
            ("elu", "ELU"),
            ("elu_0.5", "ELU_0.5"),
            ("celu", "CELU"),
            ("celu_0.1", "CELU_0.1"),
        ],
    )
    def test_mean_activation_translation(self, porter, drvi_value, expected):
        assert porter._translate_mean_activation(drvi_value) == expected

    def test_mean_activation_callable_passthrough(self, porter):
        assert porter._translate_mean_activation(torch.nn.ReLU) is torch.nn.ReLU

    def test_mean_activation_unknown_raises(self, porter):
        with pytest.raises(DRVIPortError):
            porter._translate_mean_activation("swish")

    def test_activation_fn(self, porter):
        porter.params = {"extra_encoder_kwargs": {"activation_fn": torch.nn.ELU}}
        assert porter._activation_fn() == "elu"  # known class -> clean string
        porter.params = {"extra_encoder_kwargs": {"activation_fn": torch.nn.GELU}}
        assert porter._activation_fn() is torch.nn.GELU  # other class -> passed through
        porter.params = {}
        assert porter._activation_fn() == "elu"  # drvi FCLayers default

    def test_activation_fn_encoder_decoder_mismatch_raises(self, porter):
        porter.params = {
            "extra_encoder_kwargs": {"activation_fn": torch.nn.ELU},
            "extra_decoder_kwargs": {"activation_fn": torch.nn.ReLU},
        }
        with pytest.raises(DRVIPortError, match="different hidden activations"):
            porter._activation_fn()
