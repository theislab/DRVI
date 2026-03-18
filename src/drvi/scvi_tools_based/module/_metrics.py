import torch
import numpy as np
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment


def get_optimal_matching_sum(score_matrix: np.ndarray) -> float:
    """
    Finds the optimal 1-to-1 matching to maximize the sum of scores.
    """
    n_label = score_matrix.shape[1]
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_scores = score_matrix[row_ind, col_ind]
    return matched_scores.sum() / n_label


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

    def _pairwise_mi(self, epsilon: float = 1e-8):
        if self.total_samples == 0:
            return torch.tensor(0.0, device=self.device)

        assert self.counts.sum() == self.total_samples * self.n_latent

        # 1. P(Z=b, Y_j=1): Prob of falling in bin 'b' AND being class 'j'
        p_z_y1 = self.counts / self.total_samples  # Shape: [n_latent, n_bins, n_label]
        
        # 2. P(Z=b): Prob of falling in bin 'b'. 
        # Because labels are one-hot (mutually exclusive), summing across labels gives the total bin count.
        p_z = p_z_y1.sum(dim=2, keepdim=True)  # Shape: [n_latent, n_bins, 1]
        
        # 3. P(Z=b, Y_j=0): Prob of falling in bin 'b' AND NOT being class 'j'
        p_z_y0 = p_z - p_z_y1  # Shape: [n_latent, n_bins, n_label]

        # 4. P(Y_j=1) and P(Y_j=0): Marginal probability of the classes
        # Summing across bins gives the total class probabilities
        p_y1 = p_z_y1.sum(dim=1)  # Shape: [n_latent, n_label]
        p_y0 = 1.0 - p_y1         # Shape: [n_latent, n_label]
        
        # --- Calculate Entropies ---
        # We use torch.xlogy(x, y) which safely evaluates to 0 if x = 0.
        
        # H(Z) = - sum_b P(z=b) * log(P(z=b))
        h_z = -torch.xlogy(p_z, p_z).sum(dim=1)  # Shape: [n_latent, 1]
        
        # H(Y_j) = - ( P(y=1)log(P(y=1)) + P(y=0)log(P(y=0)) )
        h_y = -(torch.xlogy(p_y1, p_y1) + torch.xlogy(p_y0, p_y0))  # Shape: [n_latent, n_label]
        
        # H(Z, Y_j) = - sum_b ( P(z=b, y=1)log(P(z=b, y=1)) + P(z=b, y=0)log(P(z=b, y=0)) )
        h_zy = -(torch.xlogy(p_z_y1, p_z_y1) + torch.xlogy(p_z_y0, p_z_y0)).sum(dim=1) # Shape: [n_latent, n_label]
        
        # --- Calculate Mutual Information ---
        # MI(Z; Y) = H(Z) + H(Y) - H(Z, Y)
        mi = h_z + h_y - h_zy
        
        # Guard against microscopic negative values caused by float32/float64 arithmetic limits
        mi = torch.clamp(mi, min=0.0) / (h_y + epsilon)
        
        return mi.detach().cpu().numpy()

    def compute(self):
        """
        Computes the One-vs-Rest (Binary) Normalized Mutual Information 
        for each latent dimension against each individual label class.
        """
        return get_optimal_matching_sum(self._pairwise_mi())
