import torch
import numpy as np
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment


def latent_matching_score(train_score_matrix: np.ndarray, val_score_matrix: np.ndarray):
    """
    Finds the optimal 1-to-1 matching to maximize the sum of scores.
    Returns the row and column indices.
    """
    cost_matrix = -train_score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_scores = val_score_matrix[row_ind, col_ind]
    return matched_scores.sum() / val_score_matrix.shape[1]


def most_similar_averaging_score(train_score_matrix: np.ndarray, val_score_matrix: np.ndarray):
    train_sorted_indices = np.argsort(-train_score_matrix, axis=0)
    top_1 = np.take_along_axis(val_score_matrix, train_sorted_indices[:1, :], axis=0)
    return top_1.mean()


def most_similar_gap_score(train_score_matrix: np.ndarray, val_score_matrix: np.ndarray):
    train_sorted_indices = np.argsort(-train_score_matrix, axis=0)
    top_2 = np.take_along_axis(val_score_matrix, train_sorted_indices[:2, :], axis=0)
    return (top_2[0] - top_2[1]).mean()


class StreamingPairwiseMI(Metric):
    full_state_update: bool = False

    def __init__(
        self,
        n_latent: int,
        n_label: int,
        n_bins: int = 10,
        dist_sync_on_step: bool = False,
    ):
        """
        Initializes the streaming contingency matrix for Pairwise MI as a torchmetric.

        Bounds used for binning come from the *previous* epoch's observed z values.
        During epoch 1 (cold start), [-5, +5] is used per latent dimension.

        At `reset()` time (between epochs), the current epoch's accumulated min/max
        is promoted to become the bounds for the next epoch.
        """
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            compute_with_cache=False,
        )

        self.n_latent = n_latent
        self.n_label = n_label
        self.n_bins = n_bins

        # Persistent buffers — the bounds adopted from the *previous* epoch.
        # Survive reset(), so they are available at the start of the next epoch.
        self.register_buffer("z_min_prev", torch.full((n_latent,), -5.0))
        self.register_buffer("z_max_prev", torch.full((n_latent,), 5.0))
        self.register_buffer("z_min_curr", torch.full((n_latent,), float('inf')))
        self.register_buffer("z_max_curr", torch.full((n_latent,), float('-inf')))

        # Epoch-level states — these ARE reset each epoch.
        # They accumulate the current epoch's observed min/max so that reset()
        # can promote them to z_min_prev / z_max_prev for the next epoch.
        # self.add_state("z_min_curr", default=torch.full((n_latent,), float('inf')), dist_reduce_fx="min")
        # self.add_state("z_max_curr", default=torch.full((n_latent,), float('-inf')), dist_reduce_fx="max")

        # Contingency matrix and sample count — reset each epoch.
        self.add_state("train_counts", default=torch.zeros((n_latent, n_bins, n_label), dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("train_total_samples", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

        self.add_state("val_counts", default=torch.zeros((n_latent, n_bins, n_label), dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("val_total_samples", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, z: torch.Tensor, label: torch.Tensor, is_train: bool):
        """
        Updates the contingency matrix with a new batch.

        Parameters
        ----------
        z: [n_batch, n_latent]  (continuous)
        label: [n_batch]  (integer class indices, already clamped to [0, n_label))
        is_train: Boolean indicating if this is a training batch.
        """
        B, L = z.shape

        # Bin using the *previous* epoch's bounds (or ±5 on the first epoch).
        z_min = self.z_min_prev
        z_max = self.z_max_prev

        range_span = (z_max - z_min).clamp(min=0.1)
        bin_width = range_span / self.n_bins

        z_binned = torch.clamp(((z - z_min) / bin_width).long(), 0, self.n_bins - 1)

        # Track running min/max for this epoch (will be saved at reset time).
        self.z_min_curr = torch.minimum(self.z_min_curr, z.min(dim=0).values.detach())
        self.z_max_curr = torch.maximum(self.z_max_curr, z.max(dim=0).values.detach())

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

    def reset_z_bounds(self):
        self.z_min_prev = self.z_min_curr.clone()
        self.z_max_prev = self.z_max_curr.clone()

    def _pairwise_mi(self, counts, total_samples, epsilon: float = 1e-8):
        if total_samples == 0:
            return np.zeros((self.n_latent, self.n_label), dtype=np.float32)

        assert counts.sum() == total_samples * self.n_latent

        # 1. P(Z=b, Y_j=1): Prob of falling in bin 'b' AND being class 'j'
        p_z_y1 = counts / total_samples  # Shape: [n_latent, n_bins, n_label]
        
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

    def compute(self, is_train: bool):
        """
        Computes the One-vs-Rest (Binary) Normalized Mutual Information 
        for each latent dimension against each individual label class.
        Finds the best matching using the training data and applies it.
        """
        train_score_matrix = self._pairwise_mi(self.train_counts, self.train_total_samples)

        if is_train:
            return {
                "LMS_SMI": latent_matching_score(train_score_matrix, train_score_matrix),
                "MSAS_SMI": most_similar_averaging_score(train_score_matrix, train_score_matrix),
                "MSGS_SMI": most_similar_gap_score(train_score_matrix, train_score_matrix)
            }
        else:
            val_score_matrix = self._pairwise_mi(self.val_counts, self.val_total_samples)
            return {
                "LMS_SMI": latent_matching_score(train_score_matrix, val_score_matrix),
                "MSAS_SMI": most_similar_averaging_score(train_score_matrix, val_score_matrix),
                "MSGS_SMI": most_similar_gap_score(train_score_matrix, val_score_matrix)
            }
