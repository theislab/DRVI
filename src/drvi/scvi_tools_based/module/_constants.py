from __future__ import annotations

# For backward compatibility
try:
    from scvi.module._constants import _MODULE_KEYS as _SCVI_MODULE_KEYS
except ImportError:
    from typing import NamedTuple

    class _NEW_SCVI_MODULE_KEYS(NamedTuple):
        X_KEY: str = "x"
        # inference
        Z_KEY: str = "z"
        QZ_KEY: str = "qz"
        QZM_KEY: str = "qzm"
        QZV_KEY: str = "qzv"
        LIBRARY_KEY: str = "library"
        QL_KEY: str = "ql"
        BATCH_INDEX_KEY: str = "batch_index"
        Y_KEY: str = "y"
        CONT_COVS_KEY: str = "cont_covs"
        CAT_COVS_KEY: str = "cat_covs"
        SIZE_FACTOR_KEY: str = "size_factor"
        # generative
        PX_KEY: str = "px"
        PL_KEY: str = "pl"
        PZ_KEY: str = "pz"
        # loss
        KL_L_KEY: str = "kl_divergence_l"
        KL_Z_KEY: str = "kl_divergence_z"

    class _SCVI_MODULE_KEYS(_NEW_SCVI_MODULE_KEYS):
        QZM_KEY: str = "qz_m"
        QZV_KEY: str = "qz_v"


class _DRVI_MODULE_KEYS(_SCVI_MODULE_KEYS):
    # generative
    PX_PARAMS_KEY = "px_params"
    PX_UNAGGREGATED_PARAMS_KEY = "px_unaggregated_params"
    # Extra
    LIKELIHOOD_ADDITIONAL_PARAMS_KEY: str = "gene_likelihood_additional_info"
    X_MASK_KEY: str = "x_mask"
    N_SAMPLES_KEY: str = "n_samples"
    RECONSTRUCTION_INDICES: str = "reconstruction_indices"
    # Tensor IO structure
    CONT_COVS_TENSOR_KEY: str = "cont_full_tensor"
    CAT_COVS_TENSOR_KEY: str = "cat_full_tensor"
    # Loss
    MSE_LOSS_KEY: str = "mse"
    RECONSTRUCTION_LOSS_KEY: str = "reconstruction_loss"


MODULE_KEYS = _DRVI_MODULE_KEYS()
