"""NN-component extensions for the internal DRVI. Developmental internal use only.

* :class:`DecoderDRVI` — decode a subset of output genes, for gene-subsampled training on very wide
  panels.
"""

from __future__ import annotations

import torch
from scvi.external.drvi import DecoderDRVI as _UpstreamDecoderDRVI
from scvi.external.drvi import StackedLinearLayer
from torch import nn
from torch.nn.functional import linear


class DecoderDRVI(_UpstreamDecoderDRVI):
    """Upstream ``DecoderDRVI`` that can restrict the parameter heads to a gene subset.

    Passing a 1-D ``reconstruction_indices`` tensor to ``forward`` runs the ``n_hidden -> n_genes``
    projection on that subset only; dense when unset. Adds no parameters. Selected via
    :attr:`DRVIModule._decoder_cls`.

    Upstream applies the parameter heads inline, so ``forward`` is reproduced here around one extra
    seam, :meth:`_apply_head`, which subclasses (:class:`~drvi.internal.sparse_drvi.SparseDecoderDRVI`)
    override to decode a subset of splits. Extra ``**kwargs`` are threaded to it, to
    :meth:`_apply_split` and to the FC body, as upstream does.
    """

    def _apply_head(self, head: nn.Module, h: torch.Tensor, reconstruction_indices=None, **kwargs) -> torch.Tensor:
        """Apply a parameter head, computing only the ``reconstruction_indices`` genes when given."""
        if reconstruction_indices is None:
            return head(h)
        if isinstance(head, StackedLinearLayer):
            return head(h, output_subset=reconstruction_indices)  # per-split head slices natively
        # shared head is a plain nn.Linear: slice its output rows before the matmul
        bias = None if head.bias is None else head.bias[reconstruction_indices]
        return linear(h, head.weight[reconstruction_indices], bias)

    def forward(self, z: torch.Tensor, *cat_list: int, cont: torch.Tensor | None = None, **kwargs):
        """Decode ``z`` as upstream does, routing the parameter heads through the seam above."""
        z_split = self._apply_split(z, **kwargs)  # (*, n_split, n_split_output)
        h = self.px_decoder(z_split, *cat_list, cont=cont, **kwargs)  # (*, n_split, n_hidden)

        px_scale_logit_per_split = self._apply_head(self.px_scale_decoder, h, **kwargs)  # (*, n_split, n_genes)
        px_scale_logit = self._aggregate(px_scale_logit_per_split)  # (*, n_genes)
        px_dropout_logit, px_r_logit = None, None
        if self.px_r_decoder is not None:
            px_r_logit = self._aggregate(self._apply_head(self.px_r_decoder, h, **kwargs))  # heuristic
        if self.px_dropout_decoder is not None:
            px_dropout_logit = -self._aggregate(-self._apply_head(self.px_dropout_decoder, h, **kwargs))  # heuristic

        if not self.inspect_mode:
            px_scale_logit_per_split = None

        return px_scale_logit, px_r_logit, px_dropout_logit, px_scale_logit_per_split
