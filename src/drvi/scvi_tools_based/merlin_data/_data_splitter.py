import lightning.pytorch as pl

from drvi.scvi_tools_based.merlin_data._data_loader import MerlinTransformedDataLoader
from drvi.scvi_tools_based.merlin_data._data_manager import MerlinDataManager


class MerlinDataSplitter(pl.LightningDataModule):
    """Creates data loaders for MerlinData object given the MerlinDataManager.

    Parameters
    ----------
    adata_manager
        MerlinDataManager object that has been created by setup_merlin_data.
    """

    data_loader_cls = MerlinTransformedDataLoader

    def __init__(
        self,
        adata_manager: MerlinDataManager,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        # We some usual inputs
        for key in ["train_size", "validation_size", "shuffle_set_split", "pin_memory"]:
            # print("Discarding key:", key)
            kwargs.pop(key, None)
        self.kwargs = kwargs

    def setup(self, stage: str | None = None):
        self.val_idx = None
        self.train_idx = None
        self.test_idx = None

    def _get_dataloader(self, split, shuffle=False, **kwargs):
        parts_per_chunk = {
            "train": 8,
            "val": 1,
            "test": 1,
        }[split]
        return self.data_loader_cls(
            self.adata_manager.get_dataset(split),
            mapping=self.adata_manager.get_fields_schema_mapping(),
            shuffle=shuffle,
            parts_per_chunk=parts_per_chunk,
            **kwargs,
        )

    def train_dataloader(self):
        """Create train data loader."""
        return self._get_dataloader("train", shuffle=True, **self.kwargs)

    def val_dataloader(self):
        """Create validation data loader."""
        return self._get_dataloader("val", shuffle=False, **self.kwargs)

    def test_dataloader(self):
        """Create test data loader."""
        return self._get_dataloader("test", shuffle=False, **self.kwargs)
