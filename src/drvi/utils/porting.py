from __future__ import annotations

import abc
import copy
import logging
import warnings
from pathlib import Path

import torch
from packaging.version import Version

logger = logging.getLogger(__name__)

__all__ = ["port_to_scvi_tools", "DRVIPorter", "DRVIPortError"]

_MODEL_FILE = "model.pt"
_NEW_ISSUE_URL = "https://github.com/theislab/drvi/issues/new"


class DRVIPortError(RuntimeError):
    """Raised when a drvi-py model uses a capability scvi-tools DRVI cannot reproduce.

    ``reason`` is a short phrase; the full user-facing message (with the issue link) is built here.
    """

    def __init__(self, reason: str):
        super().__init__(
            f"does not port: {reason}.\nIf you need to port such a model, please open an issue at "
            f"{_NEW_ISSUE_URL} and we will look into supporting it."
        )


class _Migration(abc.ABC):
    """A single version-hop transform; :meth:`apply` mutates the drvi-py ``params`` dict in place.

    Which hop each migration covers is declared by :data:`_MIGRATION_CHAIN`, not by the class.
    """

    @abc.abstractmethod
    def apply(self, params: dict) -> None: ...


class _Identity(_Migration):
    """No-op hop: the version bump needs no parameter change."""

    def apply(self, params: dict) -> None:
        pass


class _DrviPy021ToDrviPy022(_Migration):
    """Migration for drvi 0.2.2: gene_likelihood renames and prior_init_obs removal."""

    VALID_GENE_LIKELIHOOD = {"nb", "pnb", "zinb", "poisson", "normal", "normal_unit_var"}
    # old gene_likelihood -> (current name, dispersion forced by the rename).
    RENAMES = {
        "pnb_softmax": ("pnb", None),
        "poisson_orig": ("poisson", None),
        "nb_orig": ("nb", None),
        "nb": ("nb", None),
        "normal": ("normal_unit_var", None),
        "normal_v": ("normal", "gene-cell"),
        "normal_sv": ("normal", "gene"),
    }

    def apply(self, params: dict) -> None:
        gene_likelihood = params.get("gene_likelihood")
        if gene_likelihood is not None and gene_likelihood not in self.VALID_GENE_LIKELIHOOD:
            if gene_likelihood not in self.RENAMES:
                raise DRVIPortError(f"gene_likelihood '{gene_likelihood}' was dropped in drvi 0.2.2")
            params["gene_likelihood"], forced_dispersion = self.RENAMES[gene_likelihood]
            if forced_dispersion is not None:
                params["dispersion"] = forced_dispersion
        if params.pop("prior_init_obs", None) is not None:
            raise DRVIPortError("'prior_init_obs' (data-initialized prior) has no equivalent")


class _DrviPy026ToScviTools150(_Migration):
    """Migration for the drvi-py -> scvi-tools hop: split value renames (split_diag->split_mask, sum->mean)."""

    # drvi value -> scvi-tools value; a value absent from the map is not portable.
    SPLIT_METHOD = {"split_map": "split_map", "split_diag": "split_mask"}
    SPLIT_AGGREGATION = {"sum": "mean", "logsumexp": "logsumexp"}

    def apply(self, params: dict) -> None:
        for key, mapping in (("split_method", self.SPLIT_METHOD), ("split_aggregation", self.SPLIT_AGGREGATION)):
            value = params.get(key)
            if value is None:
                continue
            if value not in mapping:
                raise DRVIPortError(f"{key} '{value}'")
            params[key] = mapping[value]


# Version lineage as a chain of hops. A saved model at ``(package, version)`` is upgraded one hop at
# a time until it reaches the final scvi-tools version; each hop's migration adjusts params for that
# step. Keys are ``(from_pkg, from_version, to_pkg, to_version)``; the sole terminal key is
# ``(package, version)`` -> None. drvi-py releases go up to 0.2.6 (PyPI); DRVI ships in scvi 1.5.0.
# Add hops for future changes.
_MIGRATION_CHAIN = {
    ("drvi", "0.2.1", "drvi", "0.2.2"): _DrviPy021ToDrviPy022(),
    ("drvi", "0.2.2", "drvi", "0.2.6"): _Identity(),  # no model-kwarg changes in 0.2.2 .. 0.2.6
    ("drvi", "0.2.6", "scvi", "1.5.0"): _DrviPy026ToScviTools150(),
    ("scvi", "1.5.0"): None,  # final version; nothing to port to
}


class DRVIPorter:
    """Transforms one drvi-py DRVI checkpoint into a scvi-tools ``scvi.external.DRVI`` checkpoint.

    Pure surgery on the loaded ``model.pt`` dict: parameters and architecture are read (the
    state-dict shapes are the source of truth), unportable capabilities are rejected, and the
    state-dict, ``init_params_`` and ``registry_`` are rewritten to scvi-tools' layout.
    """

    # param -> (values scvi-tools can reproduce, reason phrase); any other value is not portable.
    REQUIRED_VALUES = {
        "covariate_modeling_strategy": ({"one_hot"}, "models categorical covariates as one-hot only"),
        "var_activation": ({"exp"}, "uses the exp variance activation only"),
        "prior": ({"normal"}, "supports the standard normal prior only"),
    }
    # param -> default; a non-default value only affects training and is dropped with a warning.
    TRAINING_ONLY = {
        "input_dropout_rate": 0.0,
        "fill_in_the_blanks_ratio": 0.0,
        "reconstruction_strategy": "dense",
        "last_layer_gradient_scale": 1.0,
    }
    # state-dict entries with no scvi-tools counterpart (param-free / training-only).
    DROPPED_STATE_PREFIXES = ("latent_stats.", "mi_metric.", "prior.", "decoder.last_layer_gradient_scaler")
    # drvi mean_activation base name -> torch.nn class name (None == identity / no-op).
    MEAN_ACTIVATION = {
        "identity": None,
        "relu": "ReLU",
        "gelu": "GELU",
        "elu": "ELU",
        "celu": "CELU",
        "leaky_relu": "LeakyReLU",
    }
    ACTIVATION_FN = {"ELU": "elu", "ReLU": "relu"}

    def __init__(self, checkpoint: dict):
        self._var_names = checkpoint["var_names"]
        self._attr = checkpoint["attr_dict"]
        self._old_sd = checkpoint["model_state_dict"]
        self._old_registry = self._attr.get("registry_", {})

        self.params = self._parse_params()
        self.arch = self._infer_architecture()
        forced_dispersion = self.params.get("dispersion")
        if forced_dispersion is not None:
            self.arch["dispersion"] = forced_dispersion
        self._check_capabilities()

        self.mean_activation = self._translate_mean_activation(self.params.get("mean_activation"))
        self.mean_activation_wrapped = self.mean_activation is not None

    @classmethod
    def from_path(cls, model_file: Path) -> DRVIPorter:
        return cls(torch.load(model_file, map_location="cpu", weights_only=False))

    def build_checkpoint(self) -> dict:
        """Assemble the scvi-tools ``model.pt`` dictionary."""
        attr = copy.deepcopy(self._attr)
        attr["init_params_"] = self._init_params()
        attr["registry_"] = self._registry()
        attr["is_trained_"] = True
        return {
            "model_state_dict": self._state_dict(),
            "var_names": self._var_names,
            "attr_dict": attr,
        }

    # -- parsing / capability checks ---------------------------------------------------------------
    def _parse_params(self) -> dict:
        """Flatten drvi ``init_params_`` and apply the version migrations."""
        init = self._attr.get("init_params_", {})
        params = dict(init.get("non_kwargs", {}))
        for inner in init.get("kwargs", {}).values():
            if isinstance(inner, dict):  # >=0.2 nests architecture args under kwargs["model_kwargs"]
                params.update(inner)
        for deprecated in ("categorical_covariates", "batch_key"):
            params.pop(deprecated, None)

        # Walk the version chain from the saved model up to the final version, applying each hop.
        pkg = "drvi"
        version = Version(self._old_registry.get("drvi_version") or "0.1.0")
        hops = sorted((k for k in _MIGRATION_CHAIN if len(k) == 4), key=lambda k: Version(k[1]))
        while hop := next((h for h in hops if h[0] == pkg and version <= Version(h[1])), None):
            _MIGRATION_CHAIN[hop].apply(params)
            pkg, version = hop[2], Version(hop[3])
        return params

    def _check_capabilities(self) -> None:
        for param, (allowed, reason) in self.REQUIRED_VALUES.items():
            value = self.params.get(param)
            if value is not None and value not in allowed:
                raise DRVIPortError(f"{param}={value!r}: scvi-tools DRVI {reason}")
        if any(".emb_list." in k or k.startswith("shared_covariate_emb") for k in self._old_sd):
            raise DRVIPortError("the model uses learned covariate embeddings")
        # scvi-tools DecoderDRVI hardcodes dropout_rate=0, so decoder dropout can't be reproduced.
        if self.params.get("decoder_dropout_rate", 0.0):
            raise DRVIPortError(f"decoder dropout (decoder_dropout_rate={self.params['decoder_dropout_rate']})")
        dropped = [k for k, default in self.TRAINING_ONLY.items() if self.params.get(k, default) != default]
        if dropped:
            warnings.warn(
                "Dropping drvi-py training-only settings (they do not change the ported model's "
                f"inference): {', '.join(dropped)}.",
                stacklevel=3,
            )

    # -- value translations ------------------------------------------------------------------------
    def _translate_mean_activation(self, value) -> str | None:
        """Translate a drvi ``mean_activation`` to a scvi-tools value; None == identity.

        drvi accepts ``"identity"``, ``"relu"``, ``"gelu"``, ``"leaky_relu[_slope]"``,
        ``"elu[_alpha]"``, ``"celu[_alpha]"``, or a callable. Strings are mapped to their torch.nn
        spelling (with the optional argument preserved); a callable/instance is passed through,
        which scvi-tools' ``_resolve_mean_activation`` accepts directly.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        for base in sorted(self.MEAN_ACTIVATION, key=len, reverse=True):  # longest first (leaky_relu)
            if value == base or value.startswith(base + "_"):
                torch_name = self.MEAN_ACTIVATION[base]
                if torch_name is None:  # identity
                    return None
                arg = value[len(base) :].lstrip("_")
                return f"{torch_name}_{arg}" if arg else torch_name
        raise DRVIPortError(f"mean_activation '{value}' has no scvi-tools equivalent")

    def _activation_fn(self):
        """Resolve the hidden-layer activation shared by scvi-tools' encoder and decoder.

        drvi keeps ``activation_fn`` (a torch.nn class, default ``nn.ELU``) in ``extra_encoder_kwargs``
        and ``extra_decoder_kwargs`` separately; scvi-tools uses a single ``activation_fn`` for both,
        so differing encoder/decoder activations cannot be reproduced. A class scvi-tools does not
        spell as a string is passed through (its ``_resolve_activation`` accepts a class directly).
        """
        resolved = {
            self._to_activation_value(self.params[key]["activation_fn"])
            for key in ("extra_encoder_kwargs", "extra_decoder_kwargs")
            if isinstance(self.params.get(key), dict) and self.params[key].get("activation_fn") is not None
        }
        if not resolved:
            return "elu"  # drvi's default hidden activation
        if len(resolved) > 1:
            raise DRVIPortError("the encoder and decoder use different hidden activations")
        return resolved.pop()

    def _to_activation_value(self, activation):
        if isinstance(activation, str):
            return activation.lower()
        return self.ACTIVATION_FN.get(getattr(activation, "__name__", None), activation)

    # -- architecture inference (state-dict shapes are the source of truth) -------------------------
    @staticmethod
    def _fc_layer_indices(state_dict: dict, prefix: str) -> list[int]:
        """Sorted layer indices ``i`` present as ``{prefix}.fc_layers.Layer {i}.0.weight``."""
        marker = f"{prefix}.fc_layers.Layer "
        return sorted(
            int(key[len(marker) : key.index(".", len(marker))])
            for key in state_dict
            if key.startswith(marker) and key.endswith(".0.weight")
        )

    def _infer_architecture(self) -> dict:
        state = self._old_sd
        enc = "z_encoder.encoder"
        enc_layers = self._fc_layer_indices(state, enc)
        if not enc_layers:
            raise DRVIPortError("unrecognized checkpoint (no encoder layers found)")

        enc_widths = [state[f"{enc}.fc_layers.Layer {i}.0.weight"].shape[0] for i in enc_layers]
        n_input = len(self._var_names)
        if state[f"{enc}.fc_layers.Layer {enc_layers[0]}.0.weight"].shape[1] != n_input:
            raise DRVIPortError("the encoder consumes covariates (encode_covariates=True)")
        n_latent = state["z_encoder.mean_encoder.fc_layers.Layer 0.0.weight"].shape[0]

        dec = "decoder.px_shared_decoder"
        dec_layers = self._fc_layer_indices(state, dec)
        dec_weights = [state[f"{dec}.fc_layers.Layer {i}.0.weight"] for i in dec_layers]
        hidden_shared = [w.dim() == 2 for w in dec_weights]  # 2-D nn.Linear shared, 3-D per-split
        dec_widths = [(w.shape[-2] if w.dim() == 3 else w.shape[0]) for w in dec_weights]

        widths = set(enc_widths + dec_widths)
        if len(widths) != 1:
            raise DRVIPortError(f"encoder/decoder layers are not all one width ({sorted(widths)})")
        if len(enc_layers) != len(dec_layers):
            raise DRVIPortError("encoder and decoder have different layer counts")

        split_weight = state.get("decoder.split_transformation_weight")  # (n_split, in, out)
        head_shared = state["decoder.params_nets.mean.fc_layers.Layer 0.0.weight"].dim() == 2
        return {
            "n_input": n_input,
            "n_latent": n_latent,
            "n_hidden": widths.pop(),
            "n_layers": len(enc_layers),
            "n_split": split_weight.shape[0] if split_weight is not None else n_latent,
            "n_split_output": split_weight.shape[2] if split_weight is not None else n_latent,
            "split_method": "split_map" if split_weight is not None else "split_diag",
            "dispersion": self._infer_dispersion(),
            "decoder_reuse_weights": self._reuse_weights(hidden_shared, head_shared),
        }

    def _infer_dispersion(self) -> str:
        state = self._old_sd
        if "decoder.params_nets.r" in state:
            return "gene"
        if "decoder.params_nets.r.weight" in state:
            return "gene-batch"
        if any(k.startswith("decoder.params_nets.r.fc_layers.") for k in state):
            return "gene-cell"
        raise DRVIPortError("could not determine dispersion from the checkpoint")

    @staticmethod
    def _reuse_weights(hidden_shared: list[bool], head_shared: bool) -> str:
        pure = {
            (True, True): "everywhere",
            (True, False): "hidden",
            (False, True): "last",
            (False, False): "nowhere",
        }
        if all(hidden_shared) or not any(hidden_shared):
            return pure[all(hidden_shared), head_shared]
        if not hidden_shared[0] and all(hidden_shared[1:]) and head_shared:
            return "hidden_except_first"
        raise DRVIPortError("the decoder weight-sharing pattern is not representable")

    # -- init_params_ / registry_ ------------------------------------------------------------------
    def _init_params(self) -> dict:
        # split_method / split_aggregation were already translated to scvi-tools' spelling by the
        # drvi-py -> scvi-tools migration; the defaults below are identical in both.
        non_kwargs = {
            "n_latent": self.arch["n_latent"],
            "n_split_latent": self.arch["n_split"],
            "split_method": self.params.get("split_method", "split_map"),
            "split_aggregation": self.params.get("split_aggregation", "logsumexp"),
            "gene_likelihood": self.params.get("gene_likelihood", "pnb"),  # drvi's default
        }
        kwargs = {
            "n_hidden": self.arch["n_hidden"],
            "n_layers": self.arch["n_layers"],
            "dispersion": self.arch["dispersion"],
            "decoder_reuse_weights": self.arch["decoder_reuse_weights"],
            "mean_activation": self.mean_activation,
            "activation_fn": self._activation_fn(),
            "use_batch_norm": self.params.get("use_batch_norm", "none"),
            "use_layer_norm": self.params.get("use_layer_norm", "both"),
            "deeply_inject_covariates": self.params.get("deeply_inject_covariates", False),
            "use_observed_lib_size": True,
            "batch_representation": "one-hot",
        }
        if self.arch["split_method"] == "split_map" and self.arch["n_split_output"] != self.arch["n_latent"]:
            kwargs["n_split_output"] = self.arch["n_split_output"]
        return {"non_kwargs": non_kwargs, "kwargs": {"kwargs": kwargs}}

    def _registry(self) -> dict:
        registry = copy.deepcopy(self._old_registry)
        registry["model_name"] = "DRVI"
        registry["setup_method_name"] = "setup_anndata"
        setup_args = dict(registry.get("setup_args", {}))
        setup_args.pop("is_count_data", None)  # drvi-only kwarg; SCVI.setup_anndata rejects it
        setup_args.setdefault("size_factor_key", None)  # SCVI reads this key unconditionally
        setup_args.setdefault("batch_key", None)
        registry["setup_args"] = setup_args
        return registry

    # -- state_dict --------------------------------------------------------------------------------
    @staticmethod
    def _transpose_per_split(weight: torch.Tensor) -> torch.Tensor:
        """Convert a per-split weight to the ``stacked_linear`` layout, leaving shared weights untouched.

        drvi's ``StackedLinearLayer`` stores ``(n_split, in, out)``; the ``stacked_linear`` package
        used by scvi-tools stores ``(n_split, out, in)``. Only per-split weights are 3-D; shared
        (2-D) weights and biases are returned unchanged.
        """
        return weight.transpose(-1, -2).contiguous() if weight.dim() == 3 else weight

    def _rename_rules(self) -> list[tuple[str, str, bool]]:
        """``(source prefix, destination prefix, transpose per-split weights)`` for direct copies."""
        mean_dst = "z_encoder.mean_encoder." + ("0." if self.mean_activation_wrapped else "")
        return [
            ("z_encoder.encoder.fc_layers.", "z_encoder.encoder.fc_layers.", False),
            ("z_encoder.mean_encoder.fc_layers.Layer 0.0.", mean_dst, False),
            ("z_encoder.var_encoder.fc_layers.Layer 0.0.", "z_encoder.var_encoder.", False),
            ("decoder.split_transformation_weight", "decoder.split_transform.weight", True),
            ("decoder.px_shared_decoder.fc_layers.", "decoder.px_decoder.fc_layers.", True),
            ("decoder.params_nets.mean.fc_layers.Layer 0.0.", "decoder.px_scale_decoder.", True),
        ]

    def _is_dropped(self, key: str) -> bool:
        return (
            key == "pyro_param_store"  # scvi's load pops this; it is not an nn parameter
            or key.startswith("decoder.params_nets.r")  # dispersion, handled by _dispersion_params
            or any(key.startswith(prefix) for prefix in self.DROPPED_STATE_PREFIXES)
        )

    def _state_dict(self) -> dict:
        new_sd = self._dispersion_params()
        rules = self._rename_rules()
        for key, value in self._old_sd.items():
            if self._is_dropped(key):
                continue
            for src, dst, transpose in rules:
                if key.startswith(src):
                    new_sd[dst + key[len(src) :]] = self._transpose_per_split(value) if transpose else value
                    break
            else:
                raise DRVIPortError(f"unexpected checkpoint parameter '{key}'")
        new_sd.update(self._library_encoder_params())
        return new_sd

    def _dispersion_params(self) -> dict:
        # drvi feeds ``r = 1.0 + param`` to its noise model; scvi-tools' generative uses the stored
        # value directly, so the +1.0 offset is folded in here.
        return {
            "gene": self._dispersion_gene,
            "gene-batch": self._dispersion_gene_batch,
            "gene-cell": self._dispersion_gene_cell,
        }[self.arch["dispersion"]]()

    def _dispersion_gene(self) -> dict:
        return {"px_r": self._old_sd["decoder.params_nets.r"] + 1.0}

    def _dispersion_gene_batch(self) -> dict:
        weight = self._old_sd["decoder.params_nets.r.weight"]  # (n_genes, n_batch)
        bias = self._old_sd["decoder.params_nets.r.bias"]  # (n_genes,)
        # (W + b) + 1 matches drvi's runtime add-order exactly (scvi does linear(one_hot, px_r)).
        return {"px_r": (weight + bias.unsqueeze(1)) + 1.0}

    def _dispersion_gene_cell(self) -> dict:
        head = "decoder.params_nets.r.fc_layers.Layer 0.0."
        return {
            "decoder.px_r_decoder.weight": self._transpose_per_split(self._old_sd[head + "weight"]),
            "decoder.px_r_decoder.bias": self._old_sd[head + "bias"] + 1.0,
        }

    def _library_encoder_params(self) -> dict:
        # scvi's VAE always has a library encoder; DRVI never reads it (observed library size), but a
        # strict load_state_dict still needs the keys. Zeros of the right shape satisfy it.
        kw = {"dtype": self._old_sd["z_encoder.encoder.fc_layers.Layer 0.0.weight"].dtype}
        n_hidden, n_input = self.arch["n_hidden"], self.arch["n_input"]
        return {
            "l_encoder.encoder.fc_layers.Layer 0.0.weight": torch.zeros(n_hidden, n_input, **kw),
            "l_encoder.encoder.fc_layers.Layer 0.0.bias": torch.zeros(n_hidden, **kw),
            "l_encoder.mean_encoder.weight": torch.zeros(1, n_hidden, **kw),
            "l_encoder.mean_encoder.bias": torch.zeros(1, **kw),
            "l_encoder.var_encoder.weight": torch.zeros(1, n_hidden, **kw),
            "l_encoder.var_encoder.bias": torch.zeros(1, **kw),
        }


def _resolve_model_file(path: Path) -> Path:
    if path.is_dir():
        return path / _MODEL_FILE
    if path.is_file():
        return path
    raise FileNotFoundError(f"No DRVI model found at {path!s}.")


def port_to_scvi_tools(
    source: str | Path,
    dest: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> str:
    """Port a drvi-py DRVI checkpoint to a scvi-tools ``scvi.external.DRVI`` checkpoint.

    Reads the drvi-py ``model.pt`` at ``source``, rewrites it, and writes a new ``model.pt`` under
    ``dest`` that can be loaded with :meth:`scvi.external.DRVI.load`. The source is never modified,
    and an existing ``dest`` is never overwritten unless ``overwrite=True``.

    No model is instantiated: this is a pure checkpoint transformation. To load the result you still
    need an ``AnnData`` whose ``var_names`` match the trained genes and whose ``obs`` has the
    covariate columns the model was set up with.

    Parameters
    ----------
    source
        Path to the drvi-py model directory (containing ``model.pt``) or the ``model.pt`` file.
    dest
        Output model directory. Defaults to ``"{source_dir}_scvi_tools"``.
    overwrite
        Overwrite ``dest`` if it already exists (default ``False``).

    Returns
    -------
    The path to the created output model directory.

    Raises
    ------
    DRVIPortError
        If the model uses a capability scvi-tools DRVI cannot reproduce.
    """
    model_file = _resolve_model_file(Path(source).expanduser())
    source_dir = model_file.parent

    dest = Path(dest).expanduser() if dest is not None else source_dir.parent / f"{source_dir.name}_scvi_tools"
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination {dest!s} already exists. Pass a different `dest` or `overwrite=True`.")

    checkpoint = DRVIPorter.from_path(model_file).build_checkpoint()
    dest.mkdir(parents=True, exist_ok=overwrite)
    torch.save(checkpoint, dest / _MODEL_FILE)
    logger.info("Ported DRVI model %s -> %s", source_dir, dest)
    return str(dest)
