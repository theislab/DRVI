"""Training plan for :class:`drvi.internal.DRVI` and the shared streaming-metrics logging mixin.

:class:`StreamingMetricsLoggingMixin` logs the module's streaming metrics at the end of each epoch
and resets them after training epochs; when the module has no streaming metrics (``latent_stats`` /
``mi_metric`` are ``None`` or absent) it is a no-op. :class:`DRVITrainingPlan` mixes it into the
upstream DRVI plan. For developmental internal use only.
"""

from __future__ import annotations

from scvi.external.drvi._trainingplan import DRVITrainingPlan as _UpstreamDRVITrainingPlan


class StreamingMetricsLoggingMixin:
    """Training-plan mixin: log the module's streaming metrics per epoch (reset after training)."""

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


class DRVITrainingPlan(StreamingMetricsLoggingMixin, _UpstreamDRVITrainingPlan):
    """Upstream scvi-tools DRVI training plan (KL annealing) plus streaming-metrics logging."""
