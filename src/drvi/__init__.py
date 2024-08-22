from importlib.metadata import version

from . import model, nn_modules, scvi_tools_based, utils

__version__ = version("drvi")

__all__ = [
    "__version__",
    "model",
    "utils",
    "nn_modules",
    "scvi_tools_based",
]
