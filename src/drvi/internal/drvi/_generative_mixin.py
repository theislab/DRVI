"""Sparse latent representation for :class:`drvi.internal.DRVI`. Developmental internal use only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from scipy import sparse
from scvi.module._constants import MODULE_KEYS

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from anndata import AnnData


class SparseLatentMixin:
    """Mixin adding sparse (thresholded) latent-representation accessors.

    Builds on the model's :meth:`~scvi.external.drvi.GenerativeMixin.iterate_on_ae_output` generator
    (run deterministically), zeros out small latent means, and returns the result as
    :class:`scipy.sparse.csr_matrix` objects — useful when a non-negative ``mean_activation`` makes
    most latent coordinates exactly (or near) zero.
    """

    @torch.inference_mode()
    def generate_sparse_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        zero_threshold: float = 0.0,
        **kwargs: Any,
    ):
        """Yield ``(sparse qz mean, sparse qz variance)`` CSR blocks, one per minibatch.

        Parameters
        ----------
        adata
            AnnData to encode. If ``None``, uses the AnnData the model was set up with.
        indices
            Indices of cells to include. If ``None``, all cells are used.
        batch_size
            Minibatch size for data loading.
        zero_threshold
            Latent means below this value are set to 0 (and the matching variances zeroed). ``0.0``
            (default) keeps the representation dense; a positive value induces sparsity.
        **kwargs
            Forwarded to ``iterate_on_ae_output``.
        """
        self._check_if_trained(warn=False)

        for inference_outputs, _ in self.iterate_on_ae_output(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            deterministic=True,
            **kwargs,
        ):
            qz = inference_outputs[MODULE_KEYS.QZ_KEY]
            qz_m = qz.loc
            qz_v = qz.variance
            if zero_threshold > 0.0:
                mask = qz_m < zero_threshold
                qz_m = qz_m.masked_fill(mask, 0.0)
                qz_v = qz_v.masked_fill(qz_m == 0.0, 0.0)
            qz_m = sparse.csr_matrix(qz_m.detach().cpu().numpy(force=True))
            qz_v = sparse.csr_matrix(qz_v.detach().cpu().numpy(force=True))
            yield qz_m, qz_v

    @torch.inference_mode()
    def get_sparse_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        zero_threshold: float = 0.0,
        return_dist: bool = False,
        **kwargs: Any,
    ):
        """Return the sparse latent representation for all requested cells.

        Parameters
        ----------
        adata, indices, batch_size, zero_threshold, **kwargs
            See :meth:`generate_sparse_latent_representation`.
        return_dist
            If ``True``, return ``(mean, variance)``; otherwise only the mean.

        Returns
        -------
        :class:`scipy.sparse.csr_matrix` or tuple of two of them.
        """
        self._check_if_trained(warn=False)

        means, variances = [], []
        for qz_m, qz_v in self.generate_sparse_latent_representation(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            zero_threshold=zero_threshold,
            **kwargs,
        ):
            means.append(qz_m)
            variances.append(qz_v)

        mean = sparse.vstack(means).tocsr()
        if return_dist:
            return mean, sparse.vstack(variances).tocsr()
        return mean
