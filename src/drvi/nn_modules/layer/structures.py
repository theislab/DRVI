from torch import nn


class SimpleResidual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)
