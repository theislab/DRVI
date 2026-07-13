from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from scvi.external import DRVI as _UpstreamDRVI

from drvi.internal._generative_mixin import SparseLatentMixin
from drvi.internal._module import DRVIModule
from drvi.internal._trainingplan import DRVITrainingPlan

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


class DRVI(_UpstreamDRVI, SparseLatentMixin):
    """Developmental DRVI model — :class:`scvi.external.DRVI` plus opt-in experimental features.

    .. warning::

        **For developmental internal use only.** This model layers experimental, unstable extras on
        top of the maintained :class:`scvi.external.DRVI`. Its API may change or be removed without
        notice; prefer :class:`scvi.external.DRVI` (aliased as :class:`drvi.model.DRVI`) for anything
        other than DRVI development.

    Everything in :class:`scvi.external.DRVI` is inherited unchanged. The additions are:

    * **Residual connections** (``residual=True``) — skip connections between the same-width hidden
      layers of the encoder and decoder bodies (``n_hidden`` is fixed across hidden layers). Needs
      ``n_layers >= 2`` to matter. See :class:`drvi.internal.DRVIModule`.
    * **Streaming (online) metrics** (``track_streaming_metrics=True``, the default) — per-epoch
      latent statistics (non-vanished dimension counts) and, when a ``labels_key`` was registered,
      streaming label/latent mutual-information scores, computed during training and logged via
      :class:`drvi.internal.DRVITrainingPlan`.
    * **Sparse latent representation** — :meth:`get_sparse_latent_representation` /
      :meth:`generate_sparse_latent_representation` from :class:`drvi.internal.SparseLatentMixin`.
    * **Gene-subsampled reconstruction** (``n_genes_to_reconstruct=N``) — reconstruct a random subset
      of ``N`` genes per training step for scalable training on very wide panels (``None`` = all
      genes). See :class:`drvi.internal.DRVIModule`.
    * **Gradient scaling** (``gradient_scale``) — scale the gradient flowing from the decoder heads
      back into the decoder body/encoder (identity forward). See :class:`drvi.internal.DRVIModule`.

    Parameters
    ----------
    adata
        AnnData registered via :meth:`~scvi.external.DRVI.setup_anndata`.
    registry
        Setup registry, to initialize without an in-memory AnnData (datamodule path).
    residual
        Enable residual connections in the encoder and decoder hidden layers.
    track_streaming_metrics
        Accumulate and log online metrics during training.
    n_genes_to_reconstruct
        ``None`` (default) reconstructs all genes; an integer ``N`` reconstructs a random subset of
        ``N`` genes per training step.
    gradient_scale
        Factor applied to the gradient from the decoder heads into the decoder body (``1.0`` = no-op).
    **kwargs
        Forwarded to :class:`scvi.external.DRVI` / :class:`drvi.internal.DRVIModule` (e.g.
        ``n_latent``, ``n_hidden``, ``n_layers``, ``split_method``, ``gene_likelihood``).

    Examples
    --------
    >>> import drvi.internal
    >>> drvi.internal.DRVI.setup_anndata(adata, batch_key="batch", labels_key="cell_type")
    >>> model = drvi.internal.DRVI(adata, n_latent=32, n_layers=2, residual=True)
    >>> model.train()
    >>> z_sparse = model.get_sparse_latent_representation(zero_threshold=0.1)
    """

    _module_cls = DRVIModule
    _training_plan_cls = DRVITrainingPlan

    def __init__(
        self,
        adata: AnnData | None = None,
        registry: dict | None = None,
        *,
        residual: bool = False,
        track_streaming_metrics: bool = True,
        n_genes_to_reconstruct: int | None = None,
        gradient_scale: float = 1.0,
        **kwargs,
    ):
        # the developmental flags flow through scvi's DRVI.__init__ **kwargs to DRVIModule.
        super().__init__(
            adata,
            registry,
            residual=residual,
            track_streaming_metrics=track_streaming_metrics,
            n_genes_to_reconstruct=n_genes_to_reconstruct,
            gradient_scale=gradient_scale,
            **kwargs,
        )
        # Recompute init_params_ against *this* subclass's signature. scvi's DRVI.__init__ builds it
        # from its own frame, but keyed on our __init__ signature (which funnels the model args
        # through **kwargs), so its explicit params (n_latent, split_method, ...) would otherwise be
        # dropped and reset to defaults on load. Capturing our own locals keeps them in ``kwargs``.
        self.init_params_ = self._get_init_params(locals())
