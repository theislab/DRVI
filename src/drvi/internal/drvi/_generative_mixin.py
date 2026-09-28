from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from scipy import sparse
from scvi.data._utils import _validate_adata_dataloader_input
from scvi.module._constants import MODULE_KEYS
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

    from anndata import AnnData
    from torch import Tensor


class SparseLatentMixin:
    """Sparse latent-representation accessors for the DRVI-family models.

    Streams the deterministic latent mean/variance blockwise and returns them as
    :class:`scipy.sparse.csr_matrix` — useful when a non-negative ``mean_activation`` or a top-k mask
    makes most latent coordinates exactly (or near) zero. Two seams let subclasses specialize:

    * :meth:`_iter_latent_mean_var` — yields per-minibatch ``(mean, var)`` tensors. The default runs
      the encoder only (:meth:`iterate_on_encoded_input`, deterministic, no decode); a subclass may
      override it.
    * :meth:`_latent_zero_mask` — the boolean mask of entries to zero at a given threshold (default
      ``|mean| < zero_threshold``).
    """

    @torch.inference_mode()
    def iterate_on_encoded_input(
        self,
        adata: AnnData | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        deterministic: bool = False,
    ):
        """Yield inference (encoder) outputs per minibatch, without running the decoder.

        Same data-loading contract as
        :meth:`~scvi.external.drvi.GenerativeMixin.iterate_on_ae_output`, but only the encoder is run
        (``module.inference``), so it is cheaper when only the latent (e.g. ``qz``) is needed. With
        ``deterministic=True`` the bottleneck uses the posterior mean.
        """
        _validate_adata_dataloader_input(self, adata, dataloader)
        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        prev_fully_deterministic = self.module.fully_deterministic
        try:
            if deterministic:
                self.module.fully_deterministic = True
            for tensors in tqdm(dataloader, mininterval=5.0):
                yield self.module.inference(**self.module._get_inference_input(tensors))
        finally:
            self.module.fully_deterministic = prev_fully_deterministic

    def _iter_latent_mean_var(
        self,
        adata: AnnData | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ):
        """Yield ``(mean, var)`` latent tensors per minibatch (encoder-only, deterministic)."""
        for inference_outputs in self.iterate_on_encoded_input(
            adata=adata, dataloader=dataloader, indices=indices, batch_size=batch_size, deterministic=True, **kwargs
        ):
            qz = inference_outputs[MODULE_KEYS.QZ_KEY]
            yield qz.loc, qz.variance

    @staticmethod
    def _latent_zero_mask(mean: Tensor, zero_threshold: float) -> Tensor:
        """Boolean mask of latent entries to zero out (default: those with ``|mean| < threshold``)."""
        return mean.abs() < zero_threshold

    @torch.inference_mode()
    def generate_sparse_latent_representation(
        self,
        adata: AnnData | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        zero_threshold: float = 0.0,
        **kwargs: Any,
    ):
        """Yield ``(sparse mean, sparse variance)`` CSR blocks, one per minibatch (streaming).

        ``zero_threshold`` (when ``> 0``) additionally zeros entries selected by
        :meth:`_latent_zero_mask`; the variance is always zeroed wherever the mean is zero.
        """
        self._check_if_trained(warn=False)
        for mean, var in self._iter_latent_mean_var(
            adata=adata, dataloader=dataloader, indices=indices, batch_size=batch_size, **kwargs
        ):
            if zero_threshold > 0.0:
                mean = mean.masked_fill(self._latent_zero_mask(mean, zero_threshold), 0.0)
            var = var.masked_fill(mean == 0.0, 0.0)
            yield (
                sparse.csr_matrix(mean.detach().cpu().numpy(force=True)),
                sparse.csr_matrix(var.detach().cpu().numpy(force=True)),
            )

    @torch.inference_mode()
    def get_sparse_latent_representation(
        self,
        adata: AnnData | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        zero_threshold: float = 0.0,
        return_dist: bool = False,
        **kwargs: Any,
    ) -> sparse.csr_matrix | tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Return the (sparse) latent mean — and optionally variance — as CSR, shape ``(n_obs, n_latent)``."""
        self._check_if_trained(warn=False)
        means, variances = [], []
        for qz_m, qz_v in self.generate_sparse_latent_representation(
            adata=adata,
            dataloader=dataloader,
            indices=indices,
            batch_size=batch_size,
            zero_threshold=zero_threshold,
            **kwargs,
        ):
            means.append(qz_m)
            variances.append(qz_v)

        mean = sparse.vstack(means, format="csr")
        if return_dist:
            return mean, sparse.vstack(variances, format="csr")
        return mean
