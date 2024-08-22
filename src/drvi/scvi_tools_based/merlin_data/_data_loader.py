import gc

import torch
from merlin.dataloader.torch import Loader


class MerlinTransformedDataLoader(Loader):
    """
    Data loader for MerlinData

    Parameters
    ----------
    mapping : list of tuple specifying (target_col, source_col)
    """

    def __init__(self, *args, mapping=None, gc_every_n_iter=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = mapping
        self.gc_every_n_iter = gc_every_n_iter
        self.iters_to_gc = gc_every_n_iter

    def convert_batch(self, batch):
        batch = super().convert_batch(batch)
        batch, _ = batch
        self.iters_to_gc -= 1
        if self.iters_to_gc < 1:
            self.iters_to_gc = self.gc_every_n_iter
            gc.collect()
        if self.mapping is None:
            return batch
        result = {}
        for new_col, col in self.mapping:
            if isinstance(col, list):
                result[new_col] = torch.stack([batch[c] for c in col], dim=1)
                for c in col:
                    del batch[c]
            else:
                result[new_col] = batch[col]
        return result
