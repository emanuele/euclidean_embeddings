"""Fastmap algorithm to obtain a k-dimensional Euclidean embedding of
a dataset of objects with a given distance function.

See: Falutsos and Lin, "FastMap: a fast algorithm for indexing,
data-mining and visualization of traditional and multimedia datasets",
1995.

"""

from __future__ import print_function
import numpy as np


cache = {}


def recursive_distance2(X, idx, distance, Y):
    """Compute the squared recursive distance between an iterable of
    objects and a single object with index (idx), given the original
    distance and the (partial) projection Y. Necessary for
    Fastmap. This is a pretty fast implementation.

    Added caching to speed-up computations.
    """
    key = (id(X), idx)
    if key in cache:
        tmp1 = cache[key]
    else:
        tmp1 = distance(X, np.array([X[idx]]))
        tmp1 *= tmp1
        cache[key] = tmp1

    tmp2 = (Y - Y[idx])
    tmp2 *= tmp2
    return tmp1.squeeze() - tmp2.sum(1)


def find_pivot_points_from_X_fast(X, distance, Y):
    """Find two points (a, b) far away, with a heuristic, from the objects
    X given the distance function.
    """
    idx_r = np.random.randint(len(X))
    idx_a = recursive_distance2(X, idx_r, distance, Y).argmax()
    idx_b = recursive_distance2(X, idx_a, distance, Y).argmax()
    return idx_a, idx_b


def find_pivot_points_scalable(X, distance, Y, k, permutation=True, c=2.0):
    """Find two points (a, b) far away, with a heuristic, from the objects
    X given the distance function, assuming objects as clustered in k
    clusters and subsampling X accordingly.
    """
    size = int(max(1, np.ceil(c * k * np.log(k))))
    if permutation:
        idx = np.random.permutation(len(X))[:size]
    else:
        idx = range(size)

    tmp_a, tmp_b = find_pivot_points_from_X_fast(X[idx], distance, Y[idx])
    return idx[tmp_a], idx[tmp_b]


def projection_from_X(X, distance, idx_a, idx_b, Y, eps=1.0e-10):
    """Compute projections of objects X, given a distance function, two
    indices of pivot points (idx_a, idx_b), and their partial
    projection Y.

    """
    # tmp1 is already computed in find_pivot_points_from_X_fast and
    # could be re-used. The caching system does that.
    tmp1 = recursive_distance2(X, idx_a, distance, Y)
    tmp2 = tmp1[idx_b] + eps
    tmp3 = recursive_distance2(X, idx_b, distance, Y)
    Yj = (tmp1 + tmp2 - tmp3) / (2.0 * np.sqrt(tmp2))
    return Yj


def compute_fastmap(X, distance, k, subsample=False, n_clusters=10,
                    verbose=False):
    """Fastmap algorithm. This is a pretty fast implementation.
    """
    Y = np.zeros([len(X), k])
    for i in range(k):
        if verbose:
            print("Dimension %s" % i)

        if subsample:
            idx_a, idx_b = find_pivot_points_scalable(X, distance, Y[:, :i], n_clusters)
        else:
            idx_a, idx_b = find_pivot_points_from_X_fast(X, distance, Y[:, :i])

        Y[:, i] = projection_from_X(X, distance, idx_a, idx_b, Y[:, :i])

    return Y
