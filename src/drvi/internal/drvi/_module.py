from __future__ import annotations

import torch
from scvi import REGISTRY_KEYS
from scvi.external.drvi import DRVIModule as _UpstreamDRVIModule
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import LossOutput, auto_move_data

from drvi.internal.drvi._base_components import DecoderDRVI
from drvi.internal.drvi._constants import DRVI_MODULE_KEYS
from drvi.internal.drvi._metrics import StreamingMetricsMixin


class DRVIModule(StreamingMetricsMixin, _UpstreamDRVIModule):
    """DRVI module with streaming metrics and gene-subsampled reconstruction.

    Adds three developmental extras on top of :class:`~scvi.external.drvi.DRVIModule`, all inherited
    unchanged otherwise:

    * ``track_streaming_metrics`` — accumulate per-batch online metrics during training
      (:class:`~drvi.internal.drvi.LatentStats`, always; :class:`~drvi.internal.drvi.StreamingPairwiseMI`,
      only when the data was set up with a ``labels_key`` so ``n_labels > 1``). Updated inside
      :meth:`loss`; logged/reset each epoch by :class:`drvi.internal.drvi.DRVITrainingPlan`.
    * ``n_genes_to_reconstruct`` — ``None`` (default) reconstructs all genes; an integer ``N``
      reconstructs a random subset of ``N`` genes per *training* step, so the decoder's
      ``n_hidden -> n_genes`` projection is done for the subset only (useful on very wide panels).
      Validation and all inference paths stay dense. See :class:`~drvi.internal.drvi.DecoderDRVI`.
    * ``gradient_scale`` — multiply the gradient flowing from the parameter heads back into the
      decoder body (and hence the encoder) by this factor; the forward pass is unchanged. ``1.0``
      (default) is a no-op.

    For developmental internal use only.
    """

    # output-subset-capable decoder (identical to upstream unless a gene subset is reconstructed).
    _decoder_cls = DecoderDRVI

    def __init__(
        self,
        *args,
        track_streaming_metrics: bool = True,
        n_genes_to_reconstruct: int | None = None,
        gradient_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # scale the gradient between the decoder heads and the decoder body (identity forward). A
        # forward hook on the body output fires on both the dense and gene-subsampled paths.
        self.gradient_scale = gradient_scale
        if gradient_scale != 1.0:
            s = gradient_scale
            self.decoder.px_decoder.register_forward_hook(
                lambda _module, _inputs, output: s * output + (1 - s) * output.detach()
            )

        # gene-subsampled reconstruction: N genes per training step (None == all genes / dense).
        # The subset path is enabled by ``_decoder_cls`` above (dense otherwise).
        self.n_genes_to_reconstruct = n_genes_to_reconstruct

        # ``n_latent`` and ``n_labels`` are set by the inherited VAE.__init__.
        self._init_streaming_metrics(track_streaming_metrics, self.n_latent, self.n_labels)

    # -- gene-subsampled reconstruction -----------------------------------------------------------
    def _get_reconstruction_indices(self, tensors: dict) -> torch.Tensor | None:
        """Random gene indices to reconstruct this step, or ``None`` for the dense path.

        Subsampling is applied only during training and only when it actually reduces the gene count.
        """
        if self.n_genes_to_reconstruct is None or not self.training:
            return None
        x = tensors[REGISTRY_KEYS.X_KEY]
        n_genes = x.shape[1]
        if self.n_genes_to_reconstruct >= n_genes:
            return None
        return torch.randperm(n_genes, device=x.device)[: self.n_genes_to_reconstruct]

    def _get_generative_input(self, tensors: dict, inference_outputs: dict) -> dict:
        gen_input = super()._get_generative_input(tensors, inference_outputs)
        reconstruction_indices = self._get_reconstruction_indices(tensors)
        gen_input[DRVI_MODULE_KEYS.RECONSTRUCTION_INDICES_KEY] = reconstruction_indices
        if reconstruction_indices is not None:
            # observed library size over the reconstructed subset (log space, as scvi expects)
            subset_lib = tensors[REGISTRY_KEYS.X_KEY][:, reconstruction_indices].sum(dim=1, keepdim=True).clamp(min=1.0)
            gen_input[MODULE_KEYS.LIBRARY_KEY] = torch.log(subset_lib)
        return gen_input

    def _compute_px_r_logit(self, px_r_logit, y, batch_index, reconstruction_indices=None, **kwargs):
        """Parent dispersion, restricted to the reconstructed gene subset when subsampling.

        Parameter-based dispersions (``gene`` / ``gene-batch`` / ``gene-label``) produce a
        full-width ``px_r``; ``gene-cell`` already comes subset from the decoder head.
        """
        px_r_logit = super()._compute_px_r_logit(px_r_logit, y, batch_index)
        if reconstruction_indices is not None and self.dispersion != "gene-cell":
            px_r_logit = px_r_logit[..., reconstruction_indices]
        return px_r_logit

    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
        reconstruction_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        """Generative step; identical to the parent unless a gene subset is being reconstructed."""
        args = (z, library, batch_index, cont_covs, cat_covs, size_factor, y, transform_batch)
        outputs = super().generative(*args, reconstruction_indices=reconstruction_indices, **kwargs)
        outputs[MODULE_KEYS.PL_KEY] = None  # subsampling uses the observed subset library size
        outputs[DRVI_MODULE_KEYS.RECONSTRUCTION_INDICES_KEY] = reconstruction_indices
        return outputs

    def loss(self, tensors: dict, inference_outputs: dict, generative_outputs: dict, **kwargs) -> LossOutput:
        """Standard DRVI loss (on the reconstructed gene subset when subsampling) plus metric updates.

        Delegates reconstruction + KL to :meth:`scvi.module.VAE.loss`; only adds the gene-subset
        target adjustment and the streaming-metric update.
        """
        reconstruction_indices = generative_outputs.get(DRVI_MODULE_KEYS.RECONSTRUCTION_INDICES_KEY)
        if reconstruction_indices is not None:
            # match the loss target to the subset the decoder produced
            tensors = {**tensors, REGISTRY_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY][:, reconstruction_indices]}
        loss_output = super().loss(tensors, inference_outputs, generative_outputs, **kwargs)
        if self.track_streaming_metrics:
            self._streaming_metrics_step(tensors, inference_outputs)
        return loss_output
