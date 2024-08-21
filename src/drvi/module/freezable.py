from torch import nn


def freezable(base_norm_class):
    class FreezableNormClass(base_norm_class):
        def __init__(self, *args, **kwargs):
            self._freeze = False
            super().__init__(*args, **kwargs)

        def freeze(self, freeze_status=True):
            self._freeze = freeze_status

        def forward(self, *args, **kwargs):
            training_status = self.training
            if self._freeze:
                self.train(False)
            result = super().forward(*args, **kwargs)
            self.train(training_status)
            return result

    return FreezableNormClass


@freezable
class FreezableBatchNorm1d(nn.BatchNorm1d):
    pass


@freezable
class FreezableLayerNorm(nn.LayerNorm):
    pass
