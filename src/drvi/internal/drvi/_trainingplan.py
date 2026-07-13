"""Training plan for :class:`drvi.internal.DRVI`.

Extends the upstream scvi-tools DRVI training plan (which anneals the KL weight over the whole run)
to log the module's streaming metrics at the end of each epoch and reset them after training epochs.
When ``track_streaming_metrics`` is off (the metrics are ``None``) it behaves exactly like the base
plan. For developmental internal use only.
"""

from __future__ import annotations

from scvi.external.drvi._trainingplan import DRVITrainingPlan as _UpstreamDRVITrainingPlan


class DRVITrainingPlan(_UpstreamDRVITrainingPlan):
    """scvi-tools DRVI training plan that also logs the module's streaming metrics per epoch."""

    def _log_streaming_metrics(self, suffix: str, is_train: bool, reset: bool) -> None:
        latent_stats = getattr(self.module, "latent_stats", None)
        mi_metric = getattr(self.module, "mi_metric", None)
        log_kwargs = {"on_step": False, "on_epoch": True, "sync_dist": self.use_sync_dist}
        if latent_stats is not None:
            self.log_dict({f"{k}_{suffix}": v for k, v in latent_stats.compute().items()}, **log_kwargs)
        if mi_metric is not None:
            metrics = mi_metric.compute(is_train=is_train)
            self.log_dict({f"{k}_{suffix}": v for k, v in metrics.items()}, **log_kwargs)
            self.log(f"lms_smi_{suffix}", metrics["LMS_SMI"], prog_bar=not is_train, **log_kwargs)
        if reset:
            for metric in (latent_stats, mi_metric):
                if metric is not None:
                    metric.reset()

    def on_train_epoch_end(self):
        # reset here (the train epoch end runs after the validation epoch end)
        self._log_streaming_metrics("train", is_train=True, reset=True)
        super().on_train_epoch_end()

    def on_validation_epoch_end(self):
        self._log_streaming_metrics("validation", is_train=False, reset=False)
        super().on_validation_epoch_end()
