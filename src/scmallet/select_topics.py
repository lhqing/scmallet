"""
See here https://stackoverflow.com/questions/52371329/fast-spearman-correlation-between-two-pandas-dataframes

Calculate correlation between two matrix, row by row
"""

import numpy as np
from numba import njit


@njit
def _mean(a):
    n = len(a)
    b = np.empty(n)
    for i in range(n):
        b[i] = a[i].mean()
    return b


@njit
def _std(a):
    n = len(a)
    b = np.empty(n)
    for i in range(n):
        b[i] = a[i].std()
    return b


@njit
def _corr(a, b, row, col):
    """Correlation between rows in a and b, no nan value."""
    _, k = a.shape

    mu_a = _mean(a)
    mu_b = _mean(b)
    sig_a = _std(a)
    sig_b = _std(b)

    out = np.zeros(shape=row.shape, dtype=np.float32)

    for idx in range(out.size):
        i = row[idx]
        j = col[idx]

        _sig_a = sig_a[i]
        _sig_b = sig_b[j]
        if _sig_a == 0 or _sig_b == 0:
            # if any variable std == 0
            out[idx] = np.nan
        else:
            out[idx] = (a[i] - mu_a[i]) @ (b[j] - mu_b[j]) / k / _sig_a / _sig_b
    return out


@njit
def _corr_all(a, b):
    """Correlation between all rows in a and b, no nan value"""
    n, k = a.shape
    m, k = b.shape

    mu_a = _mean(a)
    mu_b = _mean(b)
    sig_a = _std(a)
    sig_b = _std(b)

    out = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            _sig_a = sig_a[i]
            _sig_b = sig_b[j]
            if _sig_a == 0 or _sig_b == 0:
                # if any variable std == 0
                out[i, j] = np.nan
            else:
                out[i, j] = (a[i] - mu_a[i]) @ (b[j] - mu_b[j]) / k / _sig_a / _sig_b
    return out


@njit
def _corr_row(a, b):
    """Correlation between each row in a and b, no nan value"""
    n, k = a.shape
    m, k = b.shape

    mu_a = _mean(a)
    mu_b = _mean(b)
    sig_a = _std(a)
    sig_b = _std(b)

    out = np.empty((n,))

    for i in range(n):
        _sig_a = sig_a[i]
        _sig_b = sig_b[i]
        if _sig_a == 0 or _sig_b == 0:
            # if any variable std == 0
            out[i] = np.nan
        else:
            out[i] = (a[i] - mu_a[i]) @ (b[i] - mu_b[i]) / k / _sig_a / _sig_b
    return out


def corr_array(a, b, method="pearson"):
    """Calculate correlation between two matrix, row by row, return 2D corr values"""
    if not isinstance(a, np.ndarray):
        a = a.values
    if not isinstance(b, np.ndarray):
        b = b.values

    if method.lower()[0] == "p":
        pass
    elif method.lower()[0] == "s":
        # turn a, b in to rank matrix
        a = a.argsort(axis=1).argsort(axis=1)
        b = b.argsort(axis=1).argsort(axis=1)
    else:
        raise ValueError("Method can only be pearson or spearman")

    return _corr_all(a, b)


def corr_rows(a, b, method="pearson"):
    """Calculate correlation between the same rows in two matrix, return 1D corr values"""
    if not isinstance(a, np.ndarray):
        a = a.values
    if not isinstance(b, np.ndarray):
        b = b.values

    if method.lower()[0] == "p":
        pass
    elif method.lower()[0] == "s":
        # turn a, b in to rank matrix
        a = a.argsort(axis=1).argsort(axis=1)
        b = b.argsort(axis=1).argsort(axis=1)
    else:
        raise ValueError("Method can only be pearson or spearman")

    return _corr_row(a, b)


def dice_score(arr1, arr2):
    """Calculate the Dice score between two 1-D binary arrays."""
    intersection = np.logical_and(arr1, arr2).sum()
    dice = 2 * intersection / (arr1.sum() + arr2.sum())
    return dice


def dice_score_rows(a, b):
    """Calculate the Dice score between the same rows in two binary matrices."""
    if not isinstance(a, np.ndarray):
        a = a.values
    if not isinstance(b, np.ndarray):
        b = b.values

    return np.array([dice_score(a[i], b[i]) for i in range(a.shape[0])], dtype=np.float32)


def dice_score_array(a, b):
    """Calculate the Dice score between all pairs of rows in two binary matrices."""
    if not isinstance(a, np.ndarray):
        a = a.values
    if not isinstance(b, np.ndarray):
        b = b.values

    n = a.shape[0]
    m = b.shape[0]

    out = np.empty((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            out[i, j] = dice_score(a[i], b[j])
    return out
