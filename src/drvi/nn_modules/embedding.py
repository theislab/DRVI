import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


class FreezableEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, n_freeze_x=0, n_freeze_y=0, **kwargs):
        self._freeze_hook = None
        self.n_freeze_x = None
        self.n_freeze_y = None
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.freeze(n_freeze_x, n_freeze_y)

    def freeze(self, n_freeze_x=0, n_freeze_y=0):
        self.n_freeze_x = n_freeze_x
        self.n_freeze_y = n_freeze_y

        if self._freeze_hook is not None:
            self._freeze_hook.remove()
            self._freeze_hook = None
        if self.n_freeze_x > 0 and self.n_freeze_y > 0:
            self._freeze_hook = self.weight.register_hook(self.partial_freeze_backward_hook)

    def partial_freeze_backward_hook(self, grad):
        with torch.no_grad():
            mask = F.pad(
                torch.zeros(self.n_freeze_x, self.n_freeze_y, device=grad.device),
                (0, self.embedding_dim - self.n_freeze_y, 0, self.num_embeddings - self.n_freeze_x),
                value=1.0,
            )
            return grad * mask

    def __repr__(self):
        if self._freeze_hook is None:
            return f"Emb({self.num_embeddings}, {self.embedding_dim})"
        else:
            return f"Emb({self.num_embeddings}, {self.embedding_dim} | freeze: {self.n_freeze_x}, {self.n_freeze_y})"


class MultiEmbedding(nn.Module):
    def __init__(
        self,
        n_embedding_list: list[int],
        embedding_dim_list: list[int],
        init_method="xavier_uniform",
        normalization=None,
        **kwargs,
    ):
        super().__init__()
        assert len(n_embedding_list) == len(embedding_dim_list)

        self.emb_list = nn.ParameterList(
            [
                FreezableEmbedding(n_embedding, embedding_dim, **kwargs)
                for n_embedding, embedding_dim in zip(n_embedding_list, embedding_dim_list, strict=False)
            ]
        )
        assert normalization in [None, "l2"]
        self.normalization = normalization
        self.reset_parameters(init_method)

    def reset_parameters(self, init_method):
        # TODO: research the correct distribution considering emb_side and layer_sizes
        for emb in self.emb_list:
            if init_method is None:
                pass
            elif callable(init_method):
                init_method(emb.weight)
            elif init_method == "xavier_uniform":
                nn.init.xavier_uniform_(emb.weight)
            elif init_method == "xavier_normal":
                nn.init.xavier_normal_(emb.weight)
            elif init_method == "uniform":
                nn.init.uniform_(emb.weight, -1.0, 1.0)
            elif init_method == "normal":
                nn.init.normal_(emb.weight)
            elif init_method == "zero":
                nn.init.zeros_(emb.weight)
            elif init_method == "one":
                nn.init._no_grad_fill_(emb.weight, 1.0)
            else:
                raise NotImplementedError()

    def forward(self, index_list: torch.Tensor) -> torch.Tensor:
        assert index_list.shape[-1] == len(self.emb_list)
        emb = torch.concat([emb(index_list[..., i]) for i, emb in enumerate(self.emb_list)], dim=-1)
        if self.normalization is None:
            return emb
        elif self.normalization == "l2":
            return F.normalize(emb, p=2, dim=1)

    @classmethod
    def from_pretrained(cls, feature_embedding_instance):
        raise NotImplementedError()

    def load_weights_from_trained_module(self, other, freeze_old=False):
        assert len(self.emb_list) >= len(other.emb_list)
        if len(self.emb_list) > len(other.emb_list):
            logging.warning(f"Extending feature embedding {other} to {self} with more feature categories.")
        else:
            logging.info(f"Extending feature embedding {other} to {self}")
        for self_emb, other_emb in zip(self.emb_list, other.emb_list, strict=False):
            assert self_emb.num_embeddings >= other_emb.num_embeddings
            with torch.no_grad():
                extension_size = (
                    0,
                    self_emb.embedding_dim - other_emb.embedding_dim,
                    0,
                    self_emb.num_embeddings - other_emb.num_embeddings,
                )
                transfer_mask = F.pad(
                    torch.zeros_like(other_emb.weight.data, device=other_emb.weight.data.device),
                    extension_size,
                    value=1.0,
                )
                extended_other_emb = F.pad(other_emb.weight.data, extension_size, value=0.0)
                self_emb.weight.data = extended_other_emb + transfer_mask * self_emb.weight.data
            if freeze_old:
                self_emb.freeze(other_emb.num_embeddings, other_emb.embedding_dim)

    def freeze_top_embs(self, n_freeze_list):
        for emb, n_freeze in zip(self.emb_list, n_freeze_list, strict=False):
            emb.freeze(n_freeze, emb.embedding_dim)

    @property
    def num_embeddings(self):
        return [emb.num_embeddings for emb in self.emb_list]

    @property
    def embedding_dim(self):
        return sum(emb.embedding_dim for emb in self.emb_list)

    def __repr__(self):
        repr_text = "cat(" + ", ".join(repr(emb) for emb in self.emb_list) + ")"
        if self.normalization:
            repr_text = f"{self.normalization}({repr_text})"
        return repr_text


class FeatureEmbedding(nn.Module):
    def __init__(self, vocab_list: list[list[str]], embedding_dims: list[int], **kwargs):
        super().__init__()
        assert len(vocab_list) == len(embedding_dims)
        self.device_container = nn.Parameter(torch.tensor([]))

        self.vocab_list = vocab_list
        n_vocab_list = [len(vocab) for vocab in self.vocab_list]
        self.multi_emb = self.define_embeddings(n_vocab_list, embedding_dims, **kwargs)

        self.__index_cache = {}
        self.__vocab_map_list_cache = None

    def reset_cache(self):
        self.__index_cache = {}
        self.__vocab_map_list_cache = None

    @property
    def vocab_map_list(self):
        if self.__vocab_map_list_cache is None:
            n_vocab_list = [len(vocab) for vocab in self.vocab_list]
            self.__vocab_map_list_cache = [
                dict(zip(vocab, range(n_vocab), strict=False))
                for vocab, n_vocab in zip(self.vocab_list, n_vocab_list, strict=False)
            ]
        return self.__vocab_map_list_cache

    @staticmethod
    def define_embeddings(n_embedding_list, embedding_dims, **kwargs):
        return MultiEmbedding(n_embedding_list, embedding_dims, **kwargs)

    def reset_parameters(self, init_method):
        self.multi_emb.reset_parameters(init_method)

    def _get_index_from_sentences(self, index_sentences: np.ndarray):
        assert index_sentences.shape[-1] == len(self.vocab_map_list)
        mapping_list = map(lambda mapping: np.vectorize(lambda key: mapping[key]), self.vocab_map_list)

        indices = torch.concat(
            [torch.from_numpy(mapping(index_sentences[..., [i]])) for i, mapping in enumerate(mapping_list)], dim=-1
        )
        return indices.to(self.device_container.device)

    def forward(self, index_sentences: np.ndarray, index_cache_key=None):
        if index_cache_key is not None and index_cache_key in self.__index_cache:
            return self.multi_emb(self.__index_cache[index_cache_key])
        indices = self._get_index_from_sentences(index_sentences)
        if index_cache_key is not None:
            self.__index_cache[index_cache_key] = indices
        return self.multi_emb(indices)

    def get_extra_state(self):
        return {"vocab_list": self.vocab_list}

    def set_extra_state(self, state):
        self.vocab_list = state["vocab_list"]
        self.__vocab_map_list_cache = None

    @classmethod
    def from_numpy_array(cls, sentences_array: np.ndarray, embedding_dims: np.ndarray | list, **kwargs):
        word_list = sentences_array.transpose().tolist()
        vocab_list = [list(OrderedDict.fromkeys(words)) for words in word_list]
        return cls(vocab_list, embedding_dims, **kwargs)

    @classmethod
    def from_pandas_dataframe(cls, sentences_df: pd.DataFrame, embedding_dims: pd.DataFrame | list, **kwargs):
        if isinstance(embedding_dims, pd.DataFrame):
            assert sentences_df.columns == embedding_dims.columns
            assert len(embedding_dims) == 1
            embedding_dims = embedding_dims.loc[0].to_list()
        return cls.from_numpy_array(sentences_df.values, embedding_dims, **kwargs)

    @classmethod
    def from_pretrained(cls, feature_embedding_instance):
        raise NotImplementedError()

    def load_weights_from_trained_module(self, other, freeze_old=False):
        assert isinstance(other, self.__class__)
        assert len(self.vocab_list) >= len(other.vocab_list)

        for self_vocab, other_vocab in zip(self.vocab_list, other.vocab_list, strict=False):
            assert len(self_vocab) >= len(other_vocab)
            for self_word, other_word in zip(self_vocab, other_vocab, strict=False):
                assert self_word == other_word

        self.multi_emb.load_weights_from_trained_module(other.multi_emb, freeze_old=freeze_old)

    @property
    def embedding_dim(self):
        return self.multi_emb.embedding_dim

    def __repr__(self):
        return f"str2index -> {repr(self.multi_emb)}"
