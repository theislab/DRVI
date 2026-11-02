from scvi.external.drvi._constants import _DRVI_MODULE_KEYS


class _InternalDRVIModuleKeys(_DRVI_MODULE_KEYS):
    # generative-output key carrying the sampled gene indices from a subsampled step to loss()
    RECONSTRUCTION_INDICES_KEY: str = "reconstruction_indices"


DRVI_MODULE_KEYS = _InternalDRVIModuleKeys()
