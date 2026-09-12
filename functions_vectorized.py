import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    d = np.diag(x)
    d_wt_zeroes = d[d != 0]
    return np.prod(d_wt_zeroes)


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """

    if (len(x) != len(y)):
        return False

    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    zero_indices = np.where(x == 0)[0]
    after_zero_indices = zero_indices + 1
    valid_indices = after_zero_indices[after_zero_indices < len(x)]
    
    if len(valid_indices) == 0:
        return 0
    return np.max(x[valid_indices])


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    return np.dot(img, coefs)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    if len(x) == 0:
        return np.array([]), np.array([])

    idx = np.flatnonzero(x[1:] != x[:-1]) + 1
    idx = np.concatenate(([0], idx, [len(x)]))
    values = x[idx[:-1]]
    counts = np.diff(idx)
    
    return values, counts


def pairwise_distance(X, Y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
    XY = np.dot(X, Y.T)
    distances_sq = np.maximum(X_sq - 2 * XY + Y_sq, 0.0)
    return np.sqrt(distances_sq)
