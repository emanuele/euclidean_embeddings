"""Simple tests for Euclidean embeddings.
"""

import numpy as np
from lipschitz import compute_lipschitz
from dissimilarity import compute_dissimilarity
from fastmap import compute_fastmap
from lmds import compute_lmds
from scipy.spatial import distance_matrix
from evaluation_metrics import stress, correlation, distortion
from distances import euclidean_distance, parallel_distance_computation
from functools import partial
from time import time


def print_evaluation(X, distance, Y, subsample_size=500):
    idx = np.random.permutation(len(X))[:subsample_size]
    D_sub = distance(X[idx], X[idx])
    DY_sub = distance_matrix(Y[idx], Y[idx])
    print("  Stress = %s" % stress(D_sub.flatten(), DY_sub.flatten()))
    print("  Correlation = %s" % correlation(D_sub.flatten(), DY_sub.flatten()))
    print("  Distortion = %s" % distortion(D_sub.flatten(), DY_sub.flatten()))


if __name__ == '__main__':
    print(__doc__)
    np.random.seed(0)
    N = 100000
    d = 20
    X = np.random.uniform(size=(N, d))
    k = 14
    euclidean_distance_parallel = partial(parallel_distance_computation, distance=euclidean_distance)
    
    print("Estimating the time and quality of embedded distances vs. original distances.")

    print("")
    print("Lipschitz embedding:")
    distance = euclidean_distance
    # distance = euclidean_distance_parallel
    t0 = time()
    Y_dissimilarity, R = compute_lipschitz(X, distance, k)
    print("%s sec." % (time() - t0))
    print_evaluation(X, distance, Y_dissimilarity)

    print("")
    print("Dissimilarity Representation:")
    distance = euclidean_distance
    # distance = euclidean_distance_parallel
    t0 = time()
    Y_dissimilarity, prototype_idx = compute_dissimilarity(X, distance, k)
    print("%s sec." % (time() - t0))
    print_evaluation(X, distance, Y_dissimilarity)

    print("")
    print("Fastmap:")
    distance = euclidean_distance
    # distance = euclidean_distance_parallel
    t0 = time()
    Y_fastmap = compute_fastmap(X, distance, k)
    print("%s sec." % (time() - t0))
    print_evaluation(X, distance, Y_fastmap)

    print("")
    print("lMDS:")
    distance = euclidean_distance
    # distance = euclidean_distance_parallel
    t0 = time()
    Y_lmds = np.array(compute_lmds(X, distance, k, nl=100,
                                   landmark_policy='random'))
    print("%s sec." % (time() - t0))
    print_evaluation(X, distance, Y_lmds)
