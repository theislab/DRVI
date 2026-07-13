"""Streaming (online) metrics computed during training for :class:`drvi.internal.DRVI`.

These are torchmetrics :class:`~torchmetrics.Metric` accumulators kept on the module and updated per
batch inside ``loss()``; the training plan computes/logs/resets them at each epoch end. For
developmental internal use only.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric


def latent_matching_score(train_score_matrix: np.ndarray, val_score_matrix: np.ndarray):
    """Optimal 1-to-1 matching maximizing the summed score; returns the mean matched val score."""
    cost_matrix = -train_score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_scores = val_score_matrix[row_ind, col_ind]
    return matched_scores.sum() / val_score_matrix.shape[1]


def most_similar_averaging_score(train_score_matrix: np.ndarray, val_score_matrix: np.ndarray):
    """Mean val score of each label's top-1 latent dimension (chosen on the train matrix)."""
    train_sorted_indices = np.argsort(-train_score_matrix, axis=0)
    top_1 = np.take_along_axis(val_score_matrix, train_sorted_indices[:1, :], axis=0)
    return top_1.mean()


def most_similar_gap_score(train_score_matrix: np.ndarray, val_score_matrix: np.ndarray):
    """Mean gap between the top-1 and top-2 latent dimensions per label (chosen on the train matrix)."""
    train_sorted_indices = np.argsort(-train_score_matrix, axis=0)
    top_2 = np.take_along_axis(val_score_matrix, train_sorted_indices[:2, :], axis=0)
    return (top_2[0] - top_2[1]).mean()


class LatentStats(Metric):
    """Track per-dimension min/max of the latent mean and report non-vanished dimension counts.

    Observed bounds are accumulated over an epoch and promoted at ``reset()`` to serve as the
    reference bounds for the next epoch (also used by :class:`StreamingPairwiseMI` for binning).
    """

    full_state_update: bool = False

    def __init__(
        self,
        n_latent: int,
        vanished_threshold: float = 0.5,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            compute_with_cache=False,
        )
        self.n_latent = n_latent
        self.vanished_threshold = vanished_threshold

        # Persistent buffers — the bounds adopted from the *previous* epoch.
        # Survive reset(), so they are available at the start of the next epoch.
        self.register_buffer("z_min_prev", torch.full((n_latent,), -5.0), persistent=False)
        self.register_buffer("z_max_prev", torch.full((n_latent,), 5.0), persistent=False)

        # Epoch-level states — these ARE reset each epoch.
        self.add_state("z_min_accum", default=torch.full((n_latent,), float("inf")), dist_reduce_fx="min")
        self.add_state("z_max_accum", default=torch.full((n_latent,), float("-inf")), dist_reduce_fx="max")

    def update(self, z: torch.Tensor):
        """Update the running min/max for the current epoch."""
        self.z_min_accum = torch.minimum(self.z_min_accum, z.min(dim=0).values.detach())
        self.z_max_accum = torch.maximum(self.z_max_accum, z.max(dim=0).values.detach())

    def reset(self):
        """Promote the current epoch's min/max to be the bounds for the next epoch."""
        self.z_min_prev = self.z_min_accum.clone()
        self.z_max_prev = self.z_max_accum.clone()
        super().reset()

    def get_latest_min_max(self):
        return self.z_min_prev, self.z_max_prev

    def compute(self):
        """Compute vanished statistics based on the previous epoch's bounds."""
        non_vanished_neg = self.z_min_prev < -self.vanished_threshold
        non_vanished_pos = self.z_max_prev > self.vanished_threshold
        non_vanished = non_vanished_pos | non_vanished_neg
        return {
            "non_vanished": non_vanished.sum().item(),
            "non_vanished_pos": non_vanished_pos.sum().item(),
            "non_vanished_neg": non_vanished_neg.sum().item(),
        }


class StreamingPairwiseMI(Metric):
    """Streaming one-vs-rest normalized mutual information between each latent dim and each label.

    Maintains a contingency matrix (per split: train / validation) built by binning the latent mean
    with the previous epoch's bounds (from ``latent_stats``), and derives matching-based summary
    scores (LMS/MSAS/MSGS) at epoch end.
    """

    full_state_update: bool = False

    def __init__(
        self,
        latent_stats: LatentStats,
        n_label: int,
        n_bins: int = 10,
        dist_sync_on_step: bool = False,
        min_bin_size: float = 0.1,
        epsilon: float = 1e-8,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            compute_with_cache=False,
        )

        self.latent_stats = latent_stats
        self.n_latent = latent_stats.n_latent
        self.n_label = n_label
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.epsilon = epsilon

        # Contingency matrix and sample count — reset each epoch.
        self.add_state(
            "train_counts",
            default=torch.zeros((self.n_latent, n_bins, n_label), dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("train_total_samples", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

        self.add_state(
            "val_counts",
            default=torch.zeros((self.n_latent, n_bins, n_label), dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state("val_total_samples", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, z: torch.Tensor, label: torch.Tensor, is_train: bool):
        """Update the contingency matrix with a new batch.

        Parameters
        ----------
        z
            Continuous latent means, shape ``(n_batch, n_latent)``.
        label
            Integer class indices, shape ``(n_batch,)`` (already clamped to ``[0, n_label)``).
        is_train
            Whether this batch belongs to the training split.
        """
        B, L = z.shape

        # Bin using the *previous* epoch's bounds (or ±5 on the first epoch).
        z_min, z_max = self.latent_stats.get_latest_min_max()

        range_span = (z_max - z_min).clamp(min=self.min_bin_size)
        bin_width = range_span / self.n_bins

        z_binned = torch.clamp(((z - z_min) / bin_width).long(), 0, self.n_bins - 1)

        # Build flat indices into the 3-D counts tensor (L, n_bins, n_label)
        y_expanded = label.unsqueeze(1).expand(B, L)
        l_idx = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        flat_indices = l_idx * (self.n_bins * self.n_label) + z_binned * self.n_label + y_expanded

        flat_counts = torch.bincount(flat_indices.flatten(), minlength=L * self.n_bins * self.n_label)

        if is_train:
            self.train_total_samples += B
            self.train_counts += flat_counts.view(L, self.n_bins, self.n_label)
        else:
            self.val_total_samples += B
            self.val_counts += flat_counts.view(L, self.n_bins, self.n_label)

    def _pairwise_mi(self, counts, total_samples):
        if total_samples == 0:
            return np.zeros((self.n_latent, self.n_label), dtype=np.float32)

        assert counts.sum() == total_samples * self.n_latent

        # P(Z=b, Y_j=1): prob of falling in bin 'b' AND being class 'j'
        p_z_y1 = counts / total_samples  # [n_latent, n_bins, n_label]
        # P(Z=b): summing across (mutually exclusive) labels gives the total bin count
        p_z = p_z_y1.sum(dim=2, keepdim=True)  # [n_latent, n_bins, 1]
        # P(Z=b, Y_j=0)
        p_z_y0 = p_z - p_z_y1  # [n_latent, n_bins, n_label]
        # P(Y_j=1) and P(Y_j=0)
        p_y1 = p_z_y1.sum(dim=1)  # [n_latent, n_label]
        p_y0 = 1.0 - p_y1  # [n_latent, n_label]

        # Entropies (torch.xlogy safely evaluates to 0 when x = 0).
        h_z = -torch.xlogy(p_z, p_z).sum(dim=1)  # [n_latent, 1]
        h_y = -(torch.xlogy(p_y1, p_y1) + torch.xlogy(p_y0, p_y0))  # [n_latent, n_label]
        h_zy = -(torch.xlogy(p_z_y1, p_z_y1) + torch.xlogy(p_z_y0, p_z_y0)).sum(dim=1)  # [n_latent, n_label]

        # MI(Z; Y) = H(Z) + H(Y) - H(Z, Y), guarded and normalized by H(Y).
        mi = h_z + h_y - h_zy
        mi = torch.clamp(mi, min=0.0) / (h_y + self.epsilon)
        return mi.detach().cpu().numpy()

    def compute(self, is_train: bool):
        """One-vs-rest normalized MI summary, matched via the training-split score matrix."""
        train_score_matrix = self._pairwise_mi(self.train_counts, self.train_total_samples)

        if is_train:
            return {
                "LMS_SMI": latent_matching_score(train_score_matrix, train_score_matrix),
                "MSAS_SMI": most_similar_averaging_score(train_score_matrix, train_score_matrix),
                "MSGS_SMI": most_similar_gap_score(train_score_matrix, train_score_matrix),
            }
        val_score_matrix = self._pairwise_mi(self.val_counts, self.val_total_samples)
        return {
            "LMS_SMI": latent_matching_score(train_score_matrix, val_score_matrix),
            "MSAS_SMI": most_similar_averaging_score(train_score_matrix, val_score_matrix),
            "MSGS_SMI": most_similar_gap_score(train_score_matrix, val_score_matrix),
        }
