"""NN-component extensions for the internal DRVI. Developmental internal use only.

Two forward-only additions over the upstream scvi-tools components, both leaving weights untouched:

* :func:`enable_residual` — skip connections between the same-width hidden layers of an
  :class:`~scvi.nn.FCLayers` / :class:`~scvi.external.drvi.SplitFCLayers` instance (``n_hidden`` is
  fixed, so a hidden layer's input and output share a width).
* :class:`DecoderDRVI` — an upstream ``DecoderDRVI`` that can decode a subset of output genes, for
  gene-subsampled training on very wide panels.
"""

from __future__ import annotations

import types

import torch
from scvi.external.drvi import DecoderDRVI as _UpstreamDecoderDRVI
from scvi.external.drvi import StackedLinearLayer
from torch import nn
from torch.nn.functional import linear, one_hot


def _residual_forward(self, x, *cat_list, cont=None):
    """``FCLayers.forward`` with a skip connection around each same-width hidden block (index > 0).

    Mirrors :meth:`scvi.nn.FCLayers.forward` (covariate one-hot prep and per-inner-module application
    via ``self._apply_layer`` / ``self._apply_batch_norm``), adding ``x = x + x_in`` after every block
    whose output width matches its input. Block 0 maps the input width to ``n_hidden`` and is skipped;
    the shape guard keeps any other width mismatch safe.
    """
    cov_list = [cont] if cont is not None else []
    if len(self.n_cat_list) > len(cat_list):
        raise ValueError("nb. categorical args provided doesn't match init. params.")
    for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
        if n_cat and cat is None:
            raise ValueError("cat not provided while n_cat != 0 in init. params.")
        if n_cat > 1:  # n_cat = 1 carries no information
            cov_list.append(cat if cat.size(1) == n_cat else one_hot(cat.squeeze(-1), n_cat))

    for i, layers in enumerate(self.fc_layers):
        x_in = x
        for layer in layers:
            if layer is None:
                continue
            if isinstance(layer, nn.BatchNorm1d):
                x = self._apply_batch_norm(layer, x)
            else:
                x = self._apply_layer(layer, x, cov_list, i)
        if i > 0 and x.shape == x_in.shape:
            x = x + x_in
    return x


def enable_residual(fc_layers) -> None:
    """Add residual connections to an (Split)FCLayers instance in place (forward-only, no new params)."""
    fc_layers.forward = types.MethodType(_residual_forward, fc_layers)
    fc_layers.is_residual = True


class DecoderDRVI(_UpstreamDecoderDRVI):
    """:class:`~scvi.external.drvi.DecoderDRVI` that can decode only a subset of output genes.

    Forward-only extension. With ``output_indices=None`` it is identical to the upstream decoder; with
    a 1-D index tensor the parameter heads are evaluated for those output genes only (so the big
    ``n_hidden -> n_genes`` projection is done for the subset). Applied in place by re-binding
    ``__class__`` (no new parameters), so weights are untouched.
    """

    def forward(self, z, *cat_list, cont=None, output_indices=None):
        if output_indices is None:
            return super().forward(z, *cat_list, cont=cont)

        z_split = self._apply_split(z)  # (*, n_split, n_split_output)
        h = self.px_decoder(z_split, *cat_list, cont=cont)  # (*, n_split, n_hidden)

        px_scale_logit = self._aggregate(self._apply_head(self.px_scale_decoder, h, output_indices))
        px_r_logit = None
        if self.px_r_decoder is not None:
            px_r_logit = self._aggregate(self._apply_head(self.px_r_decoder, h, output_indices))
        px_dropout_logit = None
        if self.px_dropout_decoder is not None:
            px_dropout_logit = -self._aggregate(-self._apply_head(self.px_dropout_decoder, h, output_indices))
        # per-split params (interpretability) are not produced on the subsampled training path
        return px_scale_logit, px_r_logit, px_dropout_logit, None

    @staticmethod
    def _apply_head(head, h: torch.Tensor, output_indices: torch.Tensor) -> torch.Tensor:
        """Apply a parameter head, computing only the ``output_indices`` output genes."""
        if isinstance(head, StackedLinearLayer):
            return head(h, output_subset=output_indices)  # per-split head slices natively
        # shared head is a plain nn.Linear: slice its output rows before the matmul
        bias = None if head.bias is None else head.bias[output_indices]
        return linear(h, head.weight[output_indices], bias)
