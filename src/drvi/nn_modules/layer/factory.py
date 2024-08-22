from torch import nn

from drvi.nn_modules.layer.linear_layer import StackedLinearLayer
from drvi.nn_modules.layer.structures import SimpleResidual


class LayerFactory:
    def __init__(self, intermediate_arch="SAME", residual_preferred=False):
        assert intermediate_arch in ["SAME", "FC"]

        self.intermediate_arch = intermediate_arch
        self.residual_preferred = residual_preferred

    def _get_normal_layer(self, d_in, d_out, bias=True, **kwargs):
        raise NotImplementedError()

    def _get_stacked_layer(self, d_channel, d_in, d_out, bias=True, **kwargs):
        raise NotImplementedError()

    def get_normal_layer(self, d_in, d_out, bias=True, intermediate_layer=None, **kwargs):
        if intermediate_layer is None:
            intermediate_layer = True
        if intermediate_layer and self.intermediate_arch == "FC":
            layer = nn.Linear(d_in, d_out, bias)
        elif (not intermediate_layer) or self.intermediate_arch == "SAME":
            layer = self._get_normal_layer(d_in, d_out, bias=True, **kwargs)
        else:
            raise NotImplementedError()

        if self.residual_preferred and (d_in == d_out):
            layer = SimpleResidual(layer)
        return layer

    def get_stacked_layer(self, d_channel, d_in, d_out, bias=True, intermediate_layer=None, **kwargs):
        if intermediate_layer is None:
            intermediate_layer = True
        if intermediate_layer and self.intermediate_arch == "FC":
            layer = StackedLinearLayer(d_channel, d_in, d_out, bias)
        elif (not intermediate_layer) or self.intermediate_arch == "SAME":
            layer = self._get_stacked_layer(d_channel, d_in, d_out, bias=True, **kwargs)
        else:
            raise NotImplementedError()

        if self.residual_preferred and (d_in == d_out):
            layer = SimpleResidual(layer)
        return layer


class FCLayerFactory(LayerFactory):
    def __init__(self, intermediate_arch="SAME", residual_preferred=False):
        super().__init__(intermediate_arch=intermediate_arch, residual_preferred=residual_preferred)

    def _get_normal_layer(self, d_in, d_out, bias=True, **kwargs):
        return nn.Linear(d_in, d_out, bias=bias)

    def _get_stacked_layer(self, d_channel, d_in, d_out, bias=True, **kwargs):
        return StackedLinearLayer(d_channel, d_in, d_out, bias=bias)

    def __str__(self):
        return f"FCLayerFactory(residual_preferred={self.residual_preferred})"
