from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.cluster import entropy, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


def _check_discrete_metric_input(gt_cat_series=None, gt_one_hot=None):
    """Validate input parameters for discrete metric functions.

    Ensures that exactly one of the two input formats is provided for
    ground truth categorical data.

    Parameters
    ----------
    gt_cat_series
        Categorical series with ground truth labels.
    gt_one_hot
        One-hot encoded ground truth matrix with shape (n_samples, n_categories).

    Raises
    ------
    ValueError
        If both or neither of gt_cat_series and gt_one_hot are provided.

    Notes
    -----
    This function is used internally by all discrete metric functions to
    ensure consistent input validation. It prevents ambiguous input scenarios
    where both formats might be provided or neither is provided.
    """
    if gt_cat_series is not None and gt_one_hot is not None:
        raise ValueError("Only one of gt_cat_series or gt_one_hot should be provided.")
    if gt_cat_series is None and gt_one_hot is None:
        raise ValueError("Either gt_cat_series or gt_one_hot must be provided.")


def _get_one_hot_encoding(gt_cat_series: pd.Series) -> np.ndarray:
    """Convert categorical series to one-hot encoded matrix.

    Parameters
    ----------
    gt_cat_series
        Categorical series with ground truth labels.

    Returns
    -------
    np.ndarray
        One-hot encoded matrix with shape (n_samples, n_categories).
        Each row represents a sample, each column represents a category.

    Notes
    -----
    This function converts a pandas categorical series to a one-hot encoded
    numpy array. The categories are ordered according to the categorical
    dtype's categories attribute.

    Examples
    --------
    >>> import pandas as pd
    >>> gt_series = pd.Series(["A", "B", "A", "C"], dtype="category")
    >>> one_hot = _get_one_hot_encoding(gt_series)
    >>> print(one_hot)
    [[1 0 0]
     [0 1 0]
     [1 0 0]
     [0 0 1]]
    """
    return np.eye(len(gt_cat_series.cat.categories))[gt_cat_series.cat.codes]


def _nn_alignment_score_per_dim(var_continues: np.ndarray, gt_01: np.ndarray) -> np.ndarray:
    """Compute nearest neighbor alignment score for a single continuous variable.

    This function calculates how well the categorical ground truth labels
    align with the rightmost nearest neighbor of samples in a continuous variable.

    Parameters
    ----------
    var_continues
        Continuous variable values with shape (n_samples,).
    gt_01
        One-hot encoded ground truth matrix with shape (n_samples, n_categories).

    Returns
    -------
    np.ndarray
        Alignment scores for each category with shape (n_categories,).
        Higher values indicate better alignment between the variable and categories.

    Notes
    -----
    The score are adjusted to account for the frequency of the categories.

    **Algorithm:**
    1. Sort samples by the continuous variable
    2. For each category, compute the fraction of adjacent pairs that share
       the same category label
    3. Normalize by the expected fraction under random ordering

    **Mathematical formula:**
    ```
    alignment = (adjacent_same_category / total_adjacent - category_frequency) / (1 - category_frequency)
    ```
    """
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


def nn_alignment_score(all_vars_continues: np.ndarray, gt_cat_series=None, gt_one_hot=None) -> np.ndarray:
    """Compute nearest neighbor alignment scores for all continuous variables.

    This function calculates how well the categorical ground truth labels
    align with the rightmost nearest neighbor of samples in each continuous variable.

    Parameters
    ----------
    all_vars_continues
        Matrix of continuous variables with shape (n_samples, n_variables).
        Each column represents a different continuous variable.
    gt_cat_series
        Categorical series with ground truth labels.
    gt_one_hot
        One-hot encoded ground truth matrix with shape (n_samples, n_categories).

    Returns
    -------
    np.ndarray
        Alignment score matrix with shape (n_variables, n_categories).
        Element [i, j] represents the alignment score between variable i
        and category j. Higher values indicate better alignment.

    Notes
    -----
    The score are adjusted to account for the frequency of the categories.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simple example: 3 variables, 2 categories
    >>> all_vars = np.array([[1.0, 2.0, 0.5], [2.0, 1.0, 0.8], [3.0, 0.5, 1.2], [0.5, 3.0, 0.9]])
    >>> gt_series = pd.Series(["A", "A", "B", "B"], dtype="category")
    >>> scores = nn_alignment_score(all_vars, gt_cat_series=gt_series)
    >>> print(scores.shape)  # (3, 2)
    >>> print(scores)
    """
    _check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = _get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    result = np.zeros([n_vars, gt_01.shape[1]])
    for i in range(n_vars):
        result[i, :] = _nn_alignment_score_per_dim(all_vars_continues[:, i], gt_01)
    return result


def discrete_scaled_mutual_info_score(
    all_vars_continues: np.ndarray, gt_cat_series=None, gt_one_hot=None, n_bins=10
) -> np.ndarray:
    """Compute mutual information scores using discretized continuous variables.

    This function discretizes continuous variables into bins and then computes
    mutual information between the discretized variables and categorical ground
    truth. This approach can capture non-linear relationships that might be
    missed by linear correlation measures.

    Parameters
    ----------
    all_vars_continues
        Matrix of continuous variables with shape (n_samples, n_variables).
        Each column represents a different continuous variable.
    gt_cat_series
        Categorical series with ground truth labels.
    gt_one_hot
        One-hot encoded ground truth matrix with shape (n_samples, n_categories).
    n_bins
        Number of bins to use for discretizing continuous variables.
        More bins capture finer details but may be more sensitive to noise.

    Returns
    -------
    np.ndarray
        Mutual information score matrix with shape (n_variables, n_categories).
        Element [i, j] represents the normalized mutual information between
        discretized variable i and category j. Scores range from 0 to 1.

    Notes
    -----
    This function uses uniform binning to discretize continuous variables,
    then computes mutual information between the discretized variables and
    categorical ground truth. The scores are normalized by the entropy of
    each ground truth category.

    **Advantages over continuous mutual information:**
    The continuous mutual information is not working as expected. More info: https://github.com/scikit-learn/scikit-learn/issues/30772

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simple example: 3 variables, 2 categories
    >>> all_vars = np.array([[1.0, 2.0, 0.5], [2.0, 1.0, 0.8], [3.0, 0.5, 1.2], [0.5, 3.0, 0.9]])
    >>> gt_series = pd.Series(["A", "A", "B", "B"], dtype="category")
    >>> scores = discrete_scaled_mutual_info_score(all_vars, gt_cat_series=gt_series, n_bins=2)
    >>> print(scores.shape)  # (3, 2)
    >>> print(scores)
    """
    _check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = _get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform", random_state=123)
    all_vars_discrete = discretizer.fit_transform(all_vars_continues)

    n_vars = all_vars_discrete.shape[1]
    n_targets = gt_01.shape[1]
    result = np.zeros([n_vars, n_targets])
    for j in range(n_targets):
        h_target = entropy(gt_01[:, j])
        for i in range(n_vars):
            result[i, j] = mutual_info_score(all_vars_discrete[:, i], gt_01[:, j]) / h_target
    return result


def spearman_correlation_score(all_vars_continues: np.ndarray, gt_cat_series=None, gt_one_hot=None) -> np.ndarray:
    """Compute Spearman correlation scores between continuous variables and categories.

    This function computes the absolute Spearman correlation coefficients
    between each continuous variable and each categorical ground truth
    variable (encoded as one-hot). Spearman correlation measures monotonic
    relationships and is robust to outliers.

    Parameters
    ----------
    all_vars_continues
        Matrix of continuous variables with shape (n_samples, n_variables).
        Each column represents a different continuous variable.
    gt_cat_series
        Categorical series with ground truth labels.
    gt_one_hot
        One-hot encoded ground truth matrix with shape (n_samples, n_categories).

    Returns
    -------
    np.ndarray
        Absolute Spearman correlation matrix with shape (n_variables, n_categories).
        Element [i, j] represents the absolute Spearman correlation between
        variable i and category j. Scores range from 0 to 1.

    Notes
    -----
    Spearman correlation measures the strength and direction of monotonic
    relationships between variables. This function uses absolute values to
    focus on the strength of relationships regardless of direction.

    **Limitations:**
    Spearman correlaton is not suitable for discrete targets.
    More info: https://www.biorxiv.org/content/10.1101/2024.11.06.622266v1.full.pdf lines 985 to 989

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simple example: 3 variables, 2 categories
    >>> all_vars = np.array([[1.0, 2.0, 0.5], [2.0, 1.0, 0.8], [3.0, 0.5, 1.2], [0.5, 3.0, 0.9]])
    >>> gt_series = pd.Series(["A", "A", "B", "B"], dtype="category")
    >>> scores = spearman_correlation_score(all_vars, gt_cat_series=gt_series)
    >>> print(scores.shape)  # (3, 2)
    >>> print(scores)
    """
    _check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = _get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    result = np.abs(stats.spearmanr(all_vars_continues, gt_01).statistic[:n_vars, n_vars:])
    return result


def _maximum_mutual_information_per_pair(var_continues: np.ndarray, gt_01: np.ndarray):
    """
    Finds the exact optimal threshold x that maximizes the normalized mutual
    information I(B; A > x) / H(B) in strictly O(N) time.

    Args:
        var_continues: 1D numpy array of shape (n_samples,).
        gt_01: 2D binary numpy array of shape (n_samples, n_categories).

    Returns
    -------
        best_x: 1D numpy array of shape (n_categories,) containing optimal split thresholds.
        max_normalized_mi: 1D numpy array of shape (n_categories,) containing max normalized mutual information.
    """
    n_samples = var_continues.shape[0]
    order = var_continues.argsort()
    var_sorted = var_continues[order]
    gt_01 = gt_01[order]

    # 1. Total counts
    total_1s = np.sum(gt_01, axis=0)
    total_0s = n_samples - total_1s

    # Probability of classes in the entire dataset
    p1_total = total_1s / n_samples
    p0_total = total_0s / n_samples

    # 2. Base Entropy H(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        e1_b = np.where(p1_total > 0, p1_total * np.log2(p1_total), 0.0)
        e0_b = np.where(p0_total > 0, p0_total * np.log2(p0_total), 0.0)
    h_b = -(e1_b + e0_b)

    # 3. Cumulative sums (Prefix sums)
    left_1s = np.cumsum(gt_01, axis=0)
    left_sizes = np.arange(1, n_samples + 1)[:, None]
    left_0s = left_sizes - left_1s

    right_1s = total_1s - left_1s
    right_sizes = n_samples - left_sizes
    right_0s = total_0s - left_0s

    # 4. Probabilities for all possible splits
    # np.maximum replaces torch.clamp to prevent division by zero
    safe_left_sizes = np.maximum(left_sizes, 1)
    safe_right_sizes = np.maximum(right_sizes, 1)

    p1_left = left_1s / safe_left_sizes
    p0_left = left_0s / safe_left_sizes
    p1_right = right_1s / safe_right_sizes
    p0_right = right_0s / safe_right_sizes

    # 5. Fast Conditional Entropy Calculation H(B | A > x)
    def compute_entropy(p1, p0):
        # Suppress warnings for log2(0); np.where handles the valid outputs safely
        with np.errstate(divide="ignore", invalid="ignore"):
            e1 = np.where(p1 > 0, p1 * np.log2(p1), 0.0)
            e0 = np.where(p0 > 0, p0 * np.log2(p0), 0.0)
        return -(e1 + e0)

    h_left = compute_entropy(p1_left, p0_left)
    h_right = compute_entropy(p1_right, p0_right)

    h_cond = (left_sizes / n_samples) * h_left + (right_sizes / n_samples) * h_right

    # 6. Mask out invalid splits (cannot split between identical continuous values)
    diffs = np.diff(var_sorted)
    valid_splits = diffs > 0

    # The last element cannot be a split point
    valid_splits = np.concatenate([valid_splits, [False]])[:, None]

    h_cond = np.where(valid_splits, h_cond, np.inf)

    # 7. Find the best split
    best_idx = np.argmin(h_cond, axis=0)
    min_h_cond = np.min(h_cond, axis=0)

    best_x = np.where(
        np.isinf(min_h_cond), np.nan, (var_sorted[best_idx] + var_sorted[np.minimum(best_idx + 1, n_samples - 1)]) / 2.0
    )

    # 8. Calculate Normalized Mutual Information: I(B; A > x) / H(B)
    mutual_information = h_b - min_h_cond
    normalized_mi = np.zeros_like(h_b)

    mask = (h_b > 0) & (~np.isinf(min_h_cond))
    normalized_mi[mask] = mutual_information[mask] / h_b[mask]

    return best_x, normalized_mi


def binary_maximum_mutual_information_score(
    all_vars_continues: np.ndarray, gt_cat_series=None, gt_one_hot=None
) -> np.ndarray:
    """
    Compute the maximum mutual information score for binary ground truth.

    Args:
        all_vars_continues: Matrix of continuous variables with shape (n_samples, n_variables).
        gt_binary: Binary ground truth matrix with shape (n_samples, n_categories).

    Returns
    -------
        np.ndarray: Maximum mutual information scores with shape (n_variables, n_categories).
    """
    _check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = _get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    n_targets = gt_01.shape[1]
    result = np.zeros([n_vars, n_targets])
    for i in range(n_vars):
        best_x, normalized_mi = _maximum_mutual_information_per_pair(all_vars_continues[:, i], gt_01)
        result[i, :] = normalized_mi
    return result
