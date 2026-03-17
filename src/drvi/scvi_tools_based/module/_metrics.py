import torch
import numpy as np
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment


def get_optimal_matching_sum(score_matrix: np.ndarray) -> float:
    """
    Finds the optimal 1-to-1 matching to maximize the sum of scores.
    """
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_scores = score_matrix[row_ind, col_ind]
    return matched_scores.sum()


class StreamingPairwiseMI(Metric):
    full_state_update: bool = False

    def __init__(self, n_latent: int, n_label: int, n_bins: int = 50, dist_sync_on_step: bool = False):
        """
        Initializes the streaming contingency matrix for Pairwise MI as a torchmetric.

        Bounds used for binning come from the *previous* epoch's observed z values.
        During epoch 1 (cold start), [-5, +5] is used per latent dimension.

        At `reset()` time (between epochs), the current epoch's accumulated min/max
        is promoted to become the bounds for the next epoch.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_latent = n_latent
        self.n_label = n_label
        self.n_bins = n_bins

        # Persistent buffers — the bounds adopted from the *previous* epoch.
        # Survive reset(), so they are available at the start of the next epoch.
        self.register_buffer("z_min_prev", torch.full((n_latent,), -5.0))
        self.register_buffer("z_max_prev", torch.full((n_latent,), 5.0))

        # Epoch-level states — these ARE reset each epoch.
        # They accumulate the current epoch's observed min/max so that reset()
        # can promote them to z_min_prev / z_max_prev for the next epoch.
        self.add_state("z_min_curr", default=torch.full((n_latent,), float('inf')), dist_reduce_fx="min")
        self.add_state("z_max_curr", default=torch.full((n_latent,), float('-inf')), dist_reduce_fx="max")

        # Contingency matrix and sample count — reset each epoch.
        self.add_state("counts", default=torch.zeros((n_latent, n_bins, n_label), dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, z: torch.Tensor, label: torch.Tensor):
        """
        Updates the contingency matrix with a new batch.

        Parameters
        ----------
        z: [n_batch, n_latent]  (continuous)
        label: [n_batch]  (integer class indices, already clamped to [0, n_label))
        """
        B, L = z.shape

        # Bin using the *previous* epoch's bounds (or ±5 on the first epoch).
        z_min = self.z_min_prev
        z_max = self.z_max_prev

        range_span = torch.where(z_max - z_min == 0, torch.ones_like(z_max), z_max - z_min)
        bin_width = range_span / self.n_bins

        z_binned = torch.clamp(((z - z_min) / bin_width).long(), 0, self.n_bins - 1)

        # Track running min/max for this epoch (will be saved at reset time).
        self.z_min_curr = torch.minimum(self.z_min_curr, z.min(dim=0).values.detach())
        self.z_max_curr = torch.maximum(self.z_max_curr, z.max(dim=0).values.detach())

        self.total_samples += B

        # Build flat indices into the 3-D counts tensor (L, n_bins, n_label)
        y_expanded = label.unsqueeze(1).expand(B, L)
        l_idx = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        flat_indices = l_idx * (self.n_bins * self.n_label) + z_binned * self.n_label + y_expanded

        flat_counts = torch.bincount(flat_indices.flatten(), minlength=L * self.n_bins * self.n_label)
        self.counts += flat_counts.view(L, self.n_bins, self.n_label)

    def reset(self):
        """Promote this epoch's observed bounds before clearing the epoch states."""
        # Only promote if we actually saw at least one batch this epoch.
        if not torch.isinf(self.z_min_curr).all():
            self.z_min_prev = self.z_min_curr.clone()
            self.z_max_prev = self.z_max_curr.clone()
        super().reset()

    def compute(self, epsilon: float = 1e-8):
        """
        Computes the optimal Matching sum of normalized Mutual Information across pairs of (z, label).
        """
        if self.total_samples == 0:
            return torch.zeros(self.n_latent, device=self.device).sum()

        p_zy = self.counts / self.total_samples
        p_z = p_zy.sum(dim=2, keepdim=True)
        p_y_per_z = p_zy.sum(dim=1, keepdim=True)  # [L, 1, n_label]
        
        # p_y is the same for all latent dimensions, but we calculate it from the joint
        # to ensure consistency with p_zy. We use the first latent dimension's marginal.
        p_y = p_y_per_z[0, 0, :]  # [n_label]
        h_y = -torch.sum(p_y * torch.log(p_y + epsilon))
        
        p_z_p_y = p_z * p_y_per_z
        mask = p_zy > 0

        mi = torch.zeros_like(p_zy)
        mi[mask] = p_zy[mask] * torch.log(p_zy[mask] / (p_z_p_y[mask] + epsilon))

        # Sum over bins ONLY to get [n_latent, n_label] matrix
        mi_matrix = mi.sum(dim=1) 
        
        # Normalize the matrix by the entropy of Y
        normalized_mi_matrix = mi_matrix / (h_y + epsilon)
        
        # Apply the matching function
        return get_optimal_matching_sum(normalized_mi_matrix.detach().cpu().numpy())
