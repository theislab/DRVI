import importlib
import logging

logger = logging.getLogger(__name__)

if importlib.util.find_spec("merlin"):
    from . import fields
    from ._data import MerlinData
    from ._data_loader import MerlinTransformedDataLoader
    from ._data_manager import MerlinDataManager
    from ._data_splitter import MerlinDataSplitter
else:
    fields = None
    MerlinData = None
    MerlinTransformedDataLoader = None
    MerlinDataManager = None
    MerlinDataSplitter = None
    logger.warning("Merlin is not installed. To use merline dataloader please install it.")

__all__ = [
    "MerlinData",
    "MerlinTransformedDataLoader",
    "MerlinDataManager",
    "MerlinDataSplitter",
    "fields",
]
