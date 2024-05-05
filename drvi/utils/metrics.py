import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from sklearn.feature_selection import mutual_info_classif


def most_similar_averaging_score(result_matrix):
    return np.mean(np.max(result_matrix, axis=0))


def latent_matching_score(result_matrix):
    row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(-result_matrix - 1e-10))
    return result_matrix[row_ind, col_ind].sum() / result_matrix.shape[1]


def nn_alignment_score_per_dim(var_continues, ct_cat_series):
    order = var_continues.argsort()
    ct_cat_series = ct_cat_series[order]
    ct_01 = np.eye(len(ct_cat_series.cat.categories))[ct_cat_series.cat.codes]
    alignment = np.clip(
        (np.sum(ct_01[:-1, :] * ct_01[1:, :], axis=0) / (np.sum(ct_01, axis=0) - 1))  # fraction of cells of this type that are next to a cell of the same type
        -
        (np.sum(ct_01, axis=0) / ct_01.shape[0]),  # cancel random neighbors when CT is frequent
        0, None,
    ) / (1 - (np.sum(ct_01, axis=0) / ct_01.shape[0]))
    return alignment


def nn_alignment_score(all_vars_continues, ct_cat_series):
    n_vars = all_vars_continues.shape[1]
    result = np.zeros([n_vars, len(ct_cat_series.cat.categories)])
    for i in range(n_vars):
        result[i, :] = nn_alignment_score_per_dim(all_vars_continues[:, i], ct_cat_series)
    return result


def local_mutual_info_score_per_binary_ct(all_vars_continues, ct_binary):
    mi_score = mutual_info_classif(all_vars_continues, ct_binary)
    ct_prob = np.sum(ct_binary == 1) / ct_binary.shape[0]
    ct_entropy = stats.entropy([ct_prob, 1-ct_prob])
    return mi_score / ct_entropy


def local_mutual_info_score(all_vars_continues, ct_cat_series):
    n_vars = all_vars_continues.shape[1]
    result = np.zeros([n_vars, len(ct_cat_series.cat.categories)])
    ct_01 = np.eye(len(ct_cat_series.cat.categories))[ct_cat_series.cat.codes].T
    for j, ct in enumerate(ct_cat_series.cat.categories):
        result[:, j] = local_mutual_info_score_per_binary_ct(all_vars_continues, ct_01[j])
    return result


def global_dim_mutual_info_score(all_vars_continues, ct_cat_series):
    mi_score = mutual_info_classif(all_vars_continues, ct_cat_series)
    ct_entropy = stats.entropy(pd.Series(ct_cat_series).value_counts(normalize=True, sort=False))
    return mi_score / ct_entropy


def spearman_correlataion_score(all_vars_continues, ct_cat_series):
    n_vars = all_vars_continues.shape[1]
    ct_01 = np.eye(len(ct_cat_series.cat.categories))[ct_cat_series.cat.codes]
    result = np.abs(stats.spearmanr(all_vars_continues, ct_01).statistic[:n_vars, n_vars:])
    return result
