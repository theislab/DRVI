from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from scvi import REGISTRY_KEYS
from scvi.distributions import NegativeBinomial, Normal, Poisson, ZeroInflatedNegativeBinomial
from scvi.external.drvi import DRVIModule as _UpstreamDRVIModule
from scvi.external.drvi import LogNegativeBinomial
from scvi.external.drvi._constants import DRVI_MODULE_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import auto_move_data
from torch.nn.functional import linear, one_hot

from drvi.internal._base_components import DecoderDRVI, enable_residual
from drvi.internal._metrics import LatentStats, StreamingPairwiseMI

if TYPE_CHECKING:
    from scvi.module.base import LossOutput

# generative-output key carrying the sampled gene indices from a subsampled step to loss()
_RECON_INDICES_KEY = "reconstruction_indices"


class DRVIModule(_UpstreamDRVIModule):
    """DRVI module with optional residual layers, streaming metrics, and gene-subsampled reconstruction.

    Adds three developmental extras on top of :class:`~scvi.external.drvi.DRVIModule`, all inherited
    unchanged otherwise:

    * ``residual`` — skip connections between the same-width hidden layers of the encoder body and
      the decoder split body (see :func:`drvi.internal.enable_residual`). Requires ``n_layers >= 2``.
    * ``track_streaming_metrics`` — accumulate per-batch online metrics during training
      (:class:`~drvi.internal.LatentStats`, always; :class:`~drvi.internal.StreamingPairwiseMI`,
      only when the data was set up with a ``labels_key`` so ``n_labels > 1``). Updated inside
      :meth:`loss`; logged/reset each epoch by :class:`drvi.internal.DRVITrainingPlan`.
    * ``n_genes_to_reconstruct`` — ``None`` (default) reconstructs all genes; an integer ``N``
      reconstructs a random subset of ``N`` genes per *training* step, so the decoder's
      ``n_hidden -> n_genes`` projection is done for the subset only (useful on very wide panels).
      Validation and all inference paths stay dense. See :class:`~drvi.internal.DecoderDRVI`.
    * ``gradient_scale`` — multiply the gradient flowing from the parameter heads back into the
      decoder body (and hence the encoder) by this factor; the forward pass is unchanged. ``1.0``
      (default) is a no-op.

    For developmental internal use only.
    """

    def __init__(
        self,
        *args,
        residual: bool = False,
        track_streaming_metrics: bool = True,
        n_genes_to_reconstruct: int | None = None,
        gradient_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.residual = residual
        if residual:
            enable_residual(self.z_encoder.encoder)
            enable_residual(self.decoder.px_decoder)

        # scale the gradient between the decoder heads and the decoder body (identity forward). A
        # forward hook on the body output fires on both the dense and gene-subsampled paths.
        self.gradient_scale = gradient_scale
        if gradient_scale != 1.0:
            s = gradient_scale
            self.decoder.px_decoder.register_forward_hook(
                lambda _module, _inputs, output: s * output + (1 - s) * output.detach()
            )

        # gene-subsampled reconstruction: N genes per training step (None == all genes / dense).
        self.n_genes_to_reconstruct = n_genes_to_reconstruct
        if n_genes_to_reconstruct is not None:
            # forward-only extension of the decoder; no new parameters, so an in-place class swap.
            self.decoder.__class__ = DecoderDRVI

        # ``n_latent`` and ``n_labels`` are set by the inherited VAE.__init__.
        self.track_streaming_metrics = track_streaming_metrics
        self.latent_stats: LatentStats | None = None
        self.mi_metric: StreamingPairwiseMI | None = None
        if track_streaming_metrics:
            self.latent_stats = LatentStats(n_latent=self.n_latent)
            if self.n_labels > 1:
                self.mi_metric = StreamingPairwiseMI(latent_stats=self.latent_stats, n_label=self.n_labels)

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
        indices = self._get_reconstruction_indices(tensors)
        gen_input[_RECON_INDICES_KEY] = indices
        if indices is not None:
            # observed library size over the reconstructed subset (log space, as scvi expects)
            subset_lib = tensors[REGISTRY_KEYS.X_KEY][:, indices].sum(dim=1, keepdim=True).clamp(min=1.0)
            gen_input[MODULE_KEYS.LIBRARY_KEY] = torch.log(subset_lib)
        return gen_input

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
    ) -> dict:
        """Generative step; identical to the parent unless a gene subset is being reconstructed.

        For the dense path (``reconstruction_indices is None``) this delegates entirely to
        :meth:`scvi.external.drvi.DRVIModule.generative`. For the subsampled path it mirrors that
        method but evaluates the decoder heads, the per-gene softmax, the library size and the
        dispersion on the ``reconstruction_indices`` genes only.
        """
        if reconstruction_indices is None:
            return super().generative(z, library, batch_index, cont_covs, cat_covs, size_factor, y, transform_batch)

        idx = reconstruction_indices
        categorical_input = torch.split(cat_covs, 1, dim=1) if cat_covs is not None else ()
        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        if not self.use_size_factor_key:
            size_factor = library

        if self.batch_representation == "embedding":
            batch_rep = self.compute_embedding(REGISTRY_KEYS.BATCH_KEY, batch_index)
            decoder_cont = batch_rep if cont_covs is None else torch.cat([cont_covs, batch_rep], dim=-1)
            decoder_cats = categorical_input
        else:
            decoder_cont = cont_covs
            decoder_cats = (batch_index, *categorical_input)

        self.decoder.inspect_mode = False
        px_scale_logit, px_r_logit, px_dropout_logit, _ = self.decoder(
            z, *decoder_cats, cont=decoder_cont, output_indices=idx
        )
        px_scale_log = px_scale_logit - torch.logsumexp(px_scale_logit, dim=-1, keepdim=True)
        px_rate_log = size_factor + px_scale_log

        # dispersion logit over the subset (gene-cell already came from the subsampled head)
        if self.dispersion == "gene-label":
            px_r_logit = linear(one_hot(y.squeeze(-1), self.n_labels).float(), self.px_r)[..., idx]
        elif self.dispersion == "gene-batch":
            px_r_logit = linear(one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)[..., idx]
        elif self.dispersion == "gene":
            px_r_logit = self.px_r[idx]

        if self.gene_likelihood == "pnb":
            px = LogNegativeBinomial(log_m=px_rate_log, log_r=px_r_logit, log_scale=px_scale_log)
        elif self.gene_likelihood in ("nb", "zinb", "poisson"):
            px_scale = torch.exp(px_scale_log)
            px_rate = torch.exp(px_rate_log)
            if self.gene_likelihood == "nb":
                px = NegativeBinomial(mu=px_rate, theta=torch.exp(px_r_logit), scale=px_scale)
            elif self.gene_likelihood == "zinb":
                px = ZeroInflatedNegativeBinomial(
                    mu=px_rate, theta=torch.exp(px_r_logit), zi_logits=px_dropout_logit, scale=px_scale
                )
            else:  # poisson
                px = Poisson(rate=px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            var = torch.nan_to_num(torch.exp(px_r_logit), posinf=100.0, neginf=0.0) + 1e-8
            px = Normal(px_scale_logit, var.sqrt(), normal_mu=px_scale_logit)
        elif self.gene_likelihood == "normal_unit_var":
            px = Normal(px_scale_logit, torch.ones_like(px_scale_logit), normal_mu=px_scale_logit)
        else:
            raise ValueError(f"Unknown gene_likelihood: {self.gene_likelihood}")

        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: None,  # subsampling requires observed library size
            MODULE_KEYS.PZ_KEY: pz,
            DRVI_MODULE_KEYS.PX_UNAGGREGATED_PARAMS_KEY: None,
            _RECON_INDICES_KEY: idx,
        }

    # -- streaming metrics ------------------------------------------------------------------------
    def _streaming_metrics_step(self, tensors: dict, inference_outputs: dict) -> None:
        """Update the online metrics from one batch (called inside :meth:`loss`)."""
        if self.latent_stats is None:
            return
        z_mean = inference_outputs[MODULE_KEYS.QZ_KEY].loc
        self.latent_stats.update(z_mean)
        if self.mi_metric is not None:
            labels = tensors.get(REGISTRY_KEYS.LABELS_KEY)
            if labels is not None:
                labels_flat = torch.clamp(labels.view(-1).long(), 0, self.n_labels - 1)
                self.mi_metric.update(z_mean, labels_flat, is_train=self.training)

    def loss(self, tensors: dict, inference_outputs: dict, generative_outputs: dict, **kwargs) -> LossOutput:
        """Standard DRVI loss (on the reconstructed gene subset when subsampling) plus metric updates."""
        indices = generative_outputs.get(_RECON_INDICES_KEY)
        if indices is not None:
            # match the loss target to the subset the decoder produced
            tensors = {**tensors, REGISTRY_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY][:, indices]}
        loss_output = super().loss(tensors, inference_outputs, generative_outputs, **kwargs)
        if self.track_streaming_metrics:
            self._streaming_metrics_step(tensors, inference_outputs)
        return loss_output
