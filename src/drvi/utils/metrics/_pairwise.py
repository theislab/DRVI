import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.cluster import contingency_matrix, entropy, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


def check_discrete_metric_input(gt_cat_series=None, gt_one_hot=None):
    if gt_cat_series is not None and gt_one_hot is not None:
        raise ValueError("Only one of gt_cat_series or gt_one_hot should be provided.")
    if gt_cat_series is None and gt_one_hot is None:
        raise ValueError("Either gt_cat_series or gt_one_hot must be provided.")


def get_one_hot_encoding(gt_cat_series):
    return np.eye(len(gt_cat_series.cat.categories))[gt_cat_series.cat.codes]


def _nn_alignment_score_per_dim(var_continues, gt_01):
    order = var_continues.argsort()
    gt_01 = gt_01[order]
    alignment = np.clip(
        (
            np.sum(gt_01[:-1, :] * gt_01[1:, :], axis=0) / (np.sum(gt_01, axis=0) - 1)
        )  # fraction of cells of this type that are next to a cell of the same type
        - (np.sum(gt_01, axis=0) / gt_01.shape[0]),  # cancel random neighbors when GT (ground-truth) is frequent
        0,
        None,
    ) / (1 - (np.sum(gt_01, axis=0) / gt_01.shape[0]))
    return alignment


def nn_alignment_score(all_vars_continues, gt_cat_series=None, gt_one_hot=None):
    check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    result = np.zeros([n_vars, gt_01.shape[1]])
    for i in range(n_vars):
        result[i, :] = _nn_alignment_score_per_dim(all_vars_continues[:, i], gt_01)
    return result


def _local_mutual_info_score_per_binary_gt(all_vars_continues, gt_binary):
    mi_score = mutual_info_classif(all_vars_continues, gt_binary, n_jobs=-1)
    gt_prob = np.sum(gt_binary == 1) / gt_binary.shape[0]
    gt_entropy = stats.entropy([gt_prob, 1 - gt_prob])
    return mi_score / gt_entropy


def local_mutual_info_score(all_vars_continues, gt_cat_series=None, gt_one_hot=None):
    check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    result = np.zeros([n_vars, gt_01.shape[1]])
    for j in range(gt_01.shape[1]):
        result[:, j] = _local_mutual_info_score_per_binary_gt(all_vars_continues, gt_01[:, j])
    return result


def discrete_mutual_info_score(all_vars_continues, gt_cat_series=None, gt_one_hot=None, n_bins=10):
    check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform", random_state=123)
    all_vars_discrete = discretizer.fit_transform(all_vars_continues)

    n_vars = all_vars_discrete.shape[1]
    n_targets = gt_01.shape[1]
    result = np.zeros([n_vars, n_targets])
    for j in range(n_targets):
        h_target = entropy(gt_01[:, j])
        for i in range(n_vars):
            contingency = contingency_matrix(all_vars_discrete[:, i], gt_01[:, j], sparse=True)
            mi = mutual_info_score(None, None, contingency=contingency)
            result[i, j] = mi / h_target
    return result


def spearman_correlataion_score(all_vars_continues, gt_cat_series=None, gt_one_hot=None):
    check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    result = np.abs(stats.spearmanr(all_vars_continues, gt_01).statistic[:n_vars, n_vars:])
    return result


def global_dim_mutual_info_score(all_vars_continues, gt_cat_series):
    # This metric is not used in any analysis, but is provided for completeness.
    mi_score = mutual_info_classif(all_vars_continues, gt_cat_series)
    gt_entropy = stats.entropy(pd.Series(gt_cat_series).value_counts(normalize=True, sort=False))
    return mi_score / gt_entropy
