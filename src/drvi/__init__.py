from importlib.metadata import version

from . import internal, model, utils

__version__ = version("drvi-py")

__all__ = [
    "__version__",
    "model",
    "utils",
    "internal",
]
