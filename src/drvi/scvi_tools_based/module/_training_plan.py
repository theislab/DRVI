from scvi.train import TrainingPlan
from scvi.module.base import BaseModuleClass


class DRVITrainingPlan(TrainingPlan):
    """Custom TrainingPlan for DRVI.

    MI metrics live on the module (``module.train_mi`` / ``module.val_mi``) and
    are updated inside ``DRVIModule.loss()``. This class computes and logs the
    total epoch MI at the end of each training and validation epoch, and then
    resets the accumulators.
    """

    def __init__(self, module: BaseModuleClass, **kwargs):
        super().__init__(module=module, **kwargs)

    def on_train_epoch_end(self):
        if getattr(self.module, "train_mi", None) is not None:
            mi_val = self.module.train_mi.compute()
            self.log(
                "mi_train",
                mi_val,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.use_sync_dist,
            )
            self.module.train_mi.reset()
        super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        if getattr(self.module, "val_mi", None) is not None:
            mi_val = self.module.val_mi.compute()
            self.log(
                "mi_validation",
                mi_val,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.use_sync_dist,
            )
            self.module.val_mi.reset()
        super().on_validation_epoch_end()
