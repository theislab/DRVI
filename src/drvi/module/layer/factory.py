from collections.abc import Iterable

import numpy as np
from torch import nn

from drvi.module.dict_io import DictionaryIOModule
from drvi.module.layer.linear_layer import StackedLinearLayer
from drvi.module.layer.structures import SimpleResidual


def split_integer(n, m=None, p=None):
    """
    Splits an integer 'n' into 'm' parts or based on probabilities 'p'.
    If 'n' is a list, applies the split to each element and result `m` (or `len(p)`) lists.

    Parameters:
    - n: Integer or list of integers to be split.
    - m: Number of parts for a uniform split.
    - p: Probabilities for a non-uniform split.

    Returns:
    - List of integers representing the split, or a list of splits (of size `m` or `len(p)`) if 'n' is a list.

    Raises:
    - NotImplementedError if neither 'm' nor 'p' is provided.
    """
    if isinstance(n, Iterable):
        return list(zip(*[split_integer(d, m=m, p=p) for d in n]))
    if m is not None:
        assert p is None
        return [n // m + (i < n % m) for i in range(m)]
    elif p is not None:
        split_values = np.floor(n * np.cumsum(p) / np.sum(p) + 0.5).astype(int)
        split_values[1:] = split_values[1:] - split_values[0:-1]
        return list(split_values)
    else:
        raise NotImplementedError()


class LayerFactory:
    def __init__(self, intermediate_arch='SAME', residual_preferred=False):
        assert intermediate_arch in ['SAME', 'FC']

        self.intermediate_arch = intermediate_arch
        self.residual_preferred = residual_preferred

    def _get_normal_layer(self, d_in, d_out, bias=True, **kwargs):
        raise NotImplementedError()

    def _get_stacked_layer(self, d_channel, d_in, d_out, bias=True, **kwargs):
        raise NotImplementedError()

    def _get_emb_layer(self, l_default_features, r_default_features, l_dims, r_dims, bias=True, **kwargs):
        raise NotImplementedError()

    def get_normal_layer(self, d_in, d_out, bias=True, intermediate_layer=None, **kwargs):
        if intermediate_layer is None:
            intermediate_layer = True
        if intermediate_layer and self.intermediate_arch == 'FC':
            layer = nn.Linear(d_in, d_out, bias)
        elif (not intermediate_layer) or self.intermediate_arch == 'SAME':
            layer = self._get_normal_layer(d_in, d_out, bias=True, **kwargs)
        else:
            raise NotImplementedError()
        
        if self.residual_preferred and (d_in == d_out):
            layer = SimpleResidual(layer)
        return layer

    def get_stacked_layer(self, d_channel, d_in, d_out, bias=True, intermediate_layer=None, **kwargs):
        if intermediate_layer is None:
            intermediate_layer = True
        if intermediate_layer and self.intermediate_arch == 'FC':
            layer = StackedLinearLayer(d_channel, d_in, d_out, bias)
        elif (not intermediate_layer) or self.intermediate_arch == 'SAME':
            layer = self._get_stacked_layer(d_channel, d_in, d_out, bias=True, **kwargs)
        else:
            raise NotImplementedError()

        if self.residual_preferred and (d_in == d_out):
            layer = SimpleResidual(layer)
        return layer

    def get_emb_layer(self, l_default_features, r_default_features, l_dims, r_dims, bias=True,
                      intermediate_layer=None, **kwargs):
        if intermediate_layer is None:
            intermediate_layer = False
        assert not intermediate_layer
        layer = self._get_emb_layer(l_default_features, r_default_features, l_dims, r_dims, bias=bias, **kwargs)
        return layer


class FCLayerFactory(LayerFactory):
    def __init__(self, intermediate_arch='SAME', residual_preferred=False):
        super().__init__(intermediate_arch=intermediate_arch, residual_preferred=residual_preferred)

    def _get_normal_layer(self, d_in, d_out, bias=True, **kwargs):
        return nn.Linear(d_in, d_out, bias=bias)

    def _get_stacked_layer(self, d_channel, d_in, d_out, bias=True, **kwargs):
        return StackedLinearLayer(d_channel, d_in, d_out, bias=bias)

    def _get_emb_layer(self, l_default_features, r_default_features, l_dims, r_dims, bias=True, **kwargs):
        return DictionaryIOModule(
            nn.Linear(len(l_default_features), len(r_default_features), bias=bias),
            main_key='x_value')
    
    def __str__(self):
        return f"FCLayerFactory(residual_preferred={self.residual_preferred})"
