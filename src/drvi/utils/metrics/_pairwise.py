import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics.cluster import contingency_matrix, entropy, mutual_info_score
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


def _local_mutual_info_score_per_binary_gt(all_vars_continues: np.ndarray, gt_binary: np.ndarray) -> np.ndarray:
    """Compute scaled mutual information for a binary ground truth variable.

    Parameters
    ----------
    all_vars_continues
        Matrix of continuous variables with shape (n_samples, n_variables).
    gt_binary
        Binary ground truth labels with shape (n_samples,).

    Returns
    -------
    np.ndarray
        Scaled mutual information scores with shape (n_variables,).
        Scores are normalized by the entropy of the binary ground truth.

    Notes
    -----
    This function computes mutual information between each continuous variable
    and a binary ground truth, then normalizes by the entropy of the ground
    truth to obtain scores between 0 and 1.

    This metric is not working as expected. More info: https://github.com/scikit-learn/scikit-learn/issues/30772

    **Mathematical formula:**
    ```
    MI(all_vars_continues, gt_binary) / entropy(gt_binary)
    ```
    """
    mi_score = mutual_info_classif(all_vars_continues, gt_binary, n_jobs=-1)
    gt_prob = np.sum(gt_binary == 1) / gt_binary.shape[0]
    gt_entropy = stats.entropy([gt_prob, 1 - gt_prob])
    return mi_score / gt_entropy


def local_mutual_info_score(all_vars_continues: np.ndarray, gt_cat_series=None, gt_one_hot=None) -> np.ndarray:
    """Compute local mutual information scores for all variables and categories.

    This function calculates the scaled mutual information between each
    continuous variable and each categorical ground truth variable. The scores
    are scaled by the entropy of each ground truth category.

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
        Mutual information score matrix with shape (n_variables, n_categories).
        Element [i, j] represents the scaled mutual information between
        variable i and category j. Scores range from 0 to 1.

    Notes
    -----
    This function calculates the scaled mutual information between each
    continuous variable and each categorical ground truth variable. The scores
    are scaled by the entropy of each ground truth category.
    This metric is not working as expected. More info: https://github.com/scikit-learn/scikit-learn/issues/30772

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simple example: 3 variables, 2 categories
    >>> all_vars = np.array([[1.0, 2.0, 0.5], [2.0, 1.0, 0.8], [3.0, 0.5, 1.2], [0.5, 3.0, 0.9]])
    >>> gt_series = pd.Series(["A", "A", "B", "B"], dtype="category")
    >>> scores = local_mutual_info_score(all_vars, gt_cat_series=gt_series)
    >>> print(scores.shape)  # (3, 2)
    >>> print(scores)
    """
    _check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = _get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    result = np.zeros([n_vars, gt_01.shape[1]])
    for j in range(gt_01.shape[1]):
        result[:, j] = _local_mutual_info_score_per_binary_gt(all_vars_continues, gt_01[:, j])
    return result


def discrete_mutual_info_score(
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
    >>> scores = discrete_mutual_info_score(all_vars, gt_cat_series=gt_series, n_bins=2)
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
            contingency = contingency_matrix(all_vars_discrete[:, i], gt_01[:, j], sparse=True)
            mi = mutual_info_score(None, None, contingency=contingency)
            result[i, j] = mi / h_target
    return result


def spearman_correlataion_score(all_vars_continues: np.ndarray, gt_cat_series=None, gt_one_hot=None) -> np.ndarray:
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
    >>> scores = spearman_correlataion_score(all_vars, gt_cat_series=gt_series)
    >>> print(scores.shape)  # (3, 2)
    >>> print(scores)
    """
    _check_discrete_metric_input(gt_cat_series, gt_one_hot)
    gt_01 = _get_one_hot_encoding(gt_cat_series) if gt_cat_series is not None else gt_one_hot

    n_vars = all_vars_continues.shape[1]
    result = np.abs(stats.spearmanr(all_vars_continues, gt_01).statistic[:n_vars, n_vars:])
    return result


def global_dim_mutual_info_score(all_vars_continues: np.ndarray, gt_cat_series: pd.Series) -> np.ndarray:
    """Compute global mutual information scores for all variables with categorical ground truth.

    This function computes the normalized mutual information between each
    continuous variable and the overall categorical ground truth. Unlike
    local mutual information, this treats the ground truth as a single
    categorical variable rather than separate binary variables.

    Parameters
    ----------
    all_vars_continues
        Matrix of continuous variables with shape (n_samples, n_variables).
        Each column represents a different continuous variable.
    gt_cat_series
        Categorical series with ground truth labels.

    Returns
    -------
    np.ndarray
        Global mutual information scores with shape (n_variables,).
        Each element represents the normalized mutual information between
        a variable and the overall categorical ground truth.

    Notes
    -----
    This metric is not used in standard disentanglement analysis but is
    provided for completeness. It measures how much information each
    variable shares with the overall categorical structure, rather than
    with individual categories.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Simple example: 3 variables, 2 categories
    >>> all_vars = np.array([[1.0, 2.0, 0.5], [2.0, 1.0, 0.8], [3.0, 0.5, 1.2], [0.5, 3.0, 0.9]])
    >>> gt_series = pd.Series(["A", "A", "B", "B"], dtype="category")
    >>> scores = global_dim_mutual_info_score(all_vars, gt_series)
    >>> print(scores.shape)  # (3,)
    >>> print(scores)
    """
    # This metric is not used in any analysis, but is provided for completeness.
    mi_score = mutual_info_classif(all_vars_continues, gt_cat_series)
    gt_entropy = stats.entropy(pd.Series(gt_cat_series).value_counts(normalize=True, sort=False))
    return mi_score / gt_entropy
