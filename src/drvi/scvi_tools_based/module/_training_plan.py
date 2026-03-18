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
        if getattr(self.module, "mi_metric", None) is not None:
            mi_metrics = self.module.mi_metric.compute(is_train=True)
            self.log_dict(
                {f"{k}_train": v for k, v in mi_metrics.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.use_sync_dist,
            )
            self.log(
                "lms_smi_train",
                mi_metrics["LMS_SMI"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.use_sync_dist,
            )
            self.module.mi_metric.reset_z_bounds()  # reset after train, as validation is done first
            self.module.mi_metric.reset()
        super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        if getattr(self.module, "mi_metric", None) is not None:
            mi_metrics = self.module.mi_metric.compute(is_train=False)
            self.log_dict(
                {f"{k}_validation": v for k, v in mi_metrics.items()},
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=self.use_sync_dist,
            )
            self.log(
                "lms_smi_validation",
                mi_metrics["LMS_SMI"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=self.use_sync_dist,
            )
        super().on_validation_epoch_end()
