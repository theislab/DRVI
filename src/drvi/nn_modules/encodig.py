from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from drvi.nn_modules.embedding import FeatureEmbedding


class MultiOneHotEncoding(nn.Module):
    def __init__(self, n_embedding_list: list[int], **kwargs):
        super().__init__()
        self.n_embedding_list = n_embedding_list
        self.device_container = nn.Parameter(torch.tensor([]))

    def forward(self, index_list: torch.Tensor) -> torch.Tensor:
        assert index_list.shape[-1] == len(self.n_embedding_list)
        result = torch.concat(
            [
                F.one_hot(index_list[..., i], num_classes=n_embedding)
                for i, n_embedding in enumerate(self.n_embedding_list)
            ],
            dim=-1,
        )
        return result.to(self.device_container.device)

    def get_extra_state(self):
        return {"n_embedding_list": self.n_embedding_list}

    def set_extra_state(self, state):
        self.n_embedding_list = state["n_embedding_list"]

    @property
    def embedding_dim(self):
        return sum(self.n_embedding_list)


class FeatureOneHotEncoding(FeatureEmbedding):
    def __init__(self, vocab_list: list[list[str]], **kwargs):
        n_vocab_list = [len(vocab) for vocab in vocab_list]
        super().__init__(vocab_list, n_vocab_list, **kwargs)

    @staticmethod
    def define_embeddings(n_embedding_list, embedding_dims, **kwargs):
        for n_key, dim in zip(n_embedding_list, embedding_dims, strict=False):
            assert n_key == dim
        return MultiOneHotEncoding(n_embedding_list, **kwargs)

    @classmethod
    def from_numpy_array(cls, sentences_array: np.ndarray, **kwargs):
        word_list = sentences_array.transpose().tolist()
        vocab_list = [list(OrderedDict.fromkeys(words)) for words in word_list]
        return cls(vocab_list, **kwargs)

    @classmethod
    def from_pandas_dataframe(cls, sentences_df: pd.DataFrame, **kwargs):
        return cls.from_numpy_array(sentences_df.values)
