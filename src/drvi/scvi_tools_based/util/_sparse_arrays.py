"""Sparse array utilities (dense/sparse handling)."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _sparse_std(X: sparse.csr_matrix, axis: int = 0, ddof: int = 0) -> np.ndarray:
    """Calculate standard deviation of a sparse matrix without densifying."""
    mean_sq = np.asarray(X.power(2).mean(axis=axis)).squeeze(axis=axis)
    sq_mean = np.asarray(np.power(X.mean(axis=axis), 2)).squeeze(axis=axis)
    var = mean_sq - sq_mean
    if ddof > 0:
        n = X.shape[axis]
        var = var * (n / (n - ddof))
    return np.sqrt(np.maximum(var, 0))


def _densify_arr(arr: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    """Densify a sparse array to a dense array."""
    return arr.todense() if sparse.issparse(arr) else arr


def _flatten_to_1d(arr: np.ndarray | sparse.csr_matrix) -> np.ndarray:
    """Flatten a dense or sparse array to a 1D array."""
    return np.asarray(_densify_arr(arr)).flatten()
