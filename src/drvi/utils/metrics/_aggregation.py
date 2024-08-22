import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def most_similar_averaging_score(result_matrix):
    return np.mean(np.max(result_matrix, axis=0))


def latent_matching_score(result_matrix):
    row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(-result_matrix - 1e-10))
    return result_matrix[row_ind, col_ind].sum() / result_matrix.shape[1]


def most_similar_gap_score(result_matrix):
    sorted_values = np.sort(result_matrix, axis=0)[::-1, :]
    return np.mean(sorted_values[0, :] - sorted_values[1, :])
