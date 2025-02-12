import importlib


def get_placeholder(name, error_message, allow_init=False):
    error_message = error_message + f" Cannot use '{name}'."

    class ClassLevelGetAttrMeta(type):
        def __getattr__(cls, name):
            raise ImportError(error_message)

    class LazyNonExistingModulePlaceholder(metaclass=ClassLevelGetAttrMeta):
        def __init__(self):
            if not allow_init:
                raise ImportError(error_message)
            super().__init__()

    return LazyNonExistingModulePlaceholder


if importlib.util.find_spec("merlin"):
    from . import fields
    from ._data import MerlinData
    from ._data_loader import MerlinTransformedDataLoader
    from ._data_manager import MerlinDataManager
    from ._data_splitter import MerlinDataSplitter
else:
    error_msg = "Merlin is not installed. To use merline dataloader please install it."
    fields = get_placeholder("fields", error_msg)
    MerlinData = get_placeholder("MerlinData", error_msg)
    MerlinTransformedDataLoader = get_placeholder("MerlinTransformedDataLoader", error_msg)
    MerlinDataManager = get_placeholder("MerlinDataManager", error_msg)
    MerlinDataSplitter = get_placeholder("MerlinDataSplitter", error_msg)


__all__ = [
    "MerlinData",
    "MerlinTransformedDataLoader",
    "MerlinDataManager",
    "MerlinDataSplitter",
    "fields",
]
