import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


def most_similar_averaging_score(result_matrix: np.ndarray) -> float:
    """Compute the most similar averaging score for disentanglement evaluation.

    This function calculates the average of the maximum scores across all
    target variables. It measures how well the each of the targets can be
    captured by any of the latent dimensions.

    Parameters
    ----------
    result_matrix
        Matrix of metric scores with shape (n_dimensions, n_categories).
        Each element [i, j] represents the score for dimension i and category j.

    Returns
    -------
    float
        The average of maximum scores across all categories.
        Higher values indicate better disentanglement.

    Notes
    -----
    The function computes:
    1. For each target variable, finds the dimension with the highest score
    2. Takes the average of these maximum scores across all target variables

    This approach treats each target variable equally and measures the average
    performance of the best-matching dimensions. It's a simple but effective
    way to assess overall disentanglement quality.

    **Mathematical formula:**
    ```
    score = mean(max(result_matrix[:, j]) for j in range(n_categories))
    ```

    Examples
    --------
    >>> import numpy as np
    >>> # Example result matrix (4 dimensions, 3 categories)
    >>> result_matrix = np.array(
    ...     [
    ...         [0.9, 0.1, 0.9],
    ...         [0.1, 0.9, 0.1],
    ...         [0.1, 0.1, 0.9],
    ...         [0.1, 0.9, 0.1],
    ...     ]
    ... )
    >>> score = most_similar_averaging_score(result_matrix)
    >>> print(f"MSAS score: {score:.3f}")  # 0.933
    """
    return np.mean(np.max(result_matrix, axis=0))


def latent_matching_score(result_matrix: np.ndarray) -> float:
    """Compute the latent matching score for disentanglement evaluation.

    This function finds the optimal one-to-one matching between dimensions
    and target variables, then computes the average
    score of the matched pairs. It ensures each dimension is assigned to
    at most one target variable and vice versa.

    Parameters
    ----------
    result_matrix
        Matrix of metric scores with shape (n_dimensions, n_categories).
        Each element [i, j] represents the score for dimension i and category j.

    Returns
    -------
    float
        The average score of the optimal dimension-category matches.
        Higher values indicate better disentanglement.

    Notes
    -----
    The function uses the Hungarian algorithm (implemented as minimum weight
    bipartite matching) to find the optimal assignment between dimensions
    and target variables.

    **Mathematical formula:**
    ```
    row_ind, col_ind = max_weight_full_bipartite_matching(result_matrix)
    score = mean(result_matrix[row_ind, col_ind])
    ```

    **Advantages:**
    - Ensures each dimension is used only once
    - Mathematically rigorous evaluation of disentanglement

    Examples
    --------
    >>> import numpy as np
    >>> # Example result matrix (4 dimensions, 3 categories)
    >>> result_matrix = np.array(
    ...     [
    ...         [0.9, 0.1, 1.0],
    ...         [0.1, 0.9, 0.1],
    ...         [0.1, 0.1, 0.9],
    ...         [0.1, 0.9, 0.1],
    ...     ]
    ... )
    >>> score = latent_matching_score(result_matrix)
    >>> print(f"LMS score: {score:.3f}")  # 0.900
    """
    row_ind, col_ind = min_weight_full_bipartite_matching(csr_matrix(-result_matrix - 1e-10))
    return result_matrix[row_ind, col_ind].sum() / result_matrix.shape[1]


def most_similar_gap_score(result_matrix: np.ndarray) -> float:
    """Compute the most similar gap score for disentanglement evaluation.

    This function measures the average gap between the best and second-best
    scores for each target variable. It quantifies how clearly each target variable is
    captured by its best-matching dimension penalized by other dimensions.

    Parameters
    ----------
    result_matrix
        Matrix of metric scores with shape (n_dimensions, n_categories).
        Each element [i, j] represents the score for dimension i and category j.

    Returns
    -------
    float
        The average gap between best and second-best scores across all target variables.
        Higher values indicate clearer dimension-target variable relationships.

    Notes
    -----
    The function computes the difference between the highest and second-highest
    scores for each target variable, then averages these gaps across all target variables.

    **Algorithm steps:**
    1. Sort scores for each target variable in descending order
    2. Compute gap between first and second highest scores for each target variable
    3. Average the gaps across all target variables

    **Mathematical formula:**
    ```
    sorted_scores = sort(result_matrix, axis=0, descending=True)
    gaps = sorted_scores[0, :] - sorted_scores[1, :]
    score = mean(gaps)
    ```

    Examples
    --------
    >>> import numpy as np
    >>> # Example result matrix (4 dimensions, 3 categories)
    >>> result_matrix = np.array(
    ...     [
    ...         [0.9, 0.1, 1.0],
    ...         [0.1, 0.9, 0.1],
    ...         [0.1, 0.1, 0.9],
    ...         [0.1, 0.9, 0.1],
    ...     ]
    ... )
    >>> score = most_similar_gap_score(result_matrix)
    >>> print(f"MSGS score: {score:.3f}")  # 0.300
    """
    sorted_values = np.sort(result_matrix, axis=0)[::-1, :]
    return np.mean(sorted_values[0, :] - sorted_values[1, :])
