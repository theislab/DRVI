import math

import torch
from torch import nn


class StackedLinearLayer(nn.Module):
    __constants__ = ["n_channels", "in_features", "out_features"]
    n_channels: int
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self, n_channels: int, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.n_channels = n_channels
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((n_channels, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(n_channels, out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self._init_weight()
        if self.bias is not None:
            self._init_bias()

    def _init_weight(self) -> None:
        # Same as default nn.Linear (https://github.com/pytorch/pytorch/issues/57109)
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.weight, -bound, bound)

    def _init_bias(self) -> None:
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mm = torch.einsum("bci,cio->bco", input, self.weight)
        if self.bias is not None:
            mm = mm + self.bias  # They will broadcast well
        return mm

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"n_channels={self.n_channels}, bias={self.bias is not None}"
        )
