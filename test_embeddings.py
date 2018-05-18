"""Simple tests for Euclidean embeddings.
"""

import numpy as np
from dissimilarity import compute_dissimilarity
from fastmap import fastmap
from lmds import compute_lmds
from scipy.spatial import distance_matrix
from evaluation_metrics import stress, correlation, distortion
from distances import euclidean_distance, parallel_distance_computation
from functools import partial
from time import time


def print_evaluation(X, Y, subsample_size=500):
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

    print("Dissimilarity Representation:")
    distance = euclidean_distance
    # distance = euclidean_distance_parallel
    t0 = time()
    Y_dissimilarity, prototype_idx = compute_dissimilarity(X, num_prototypes=k,
                                                           distance=distance,
                                                           prototype_policy='fft',
                                                           verbose=False)
    print("%s sec." % (time() - t0))
    print_evaluation(X, Y_dissimilarity)

    print("Fastmap:")
    distance = euclidean_distance
    # distance = euclidean_distance_parallel
    t0 = time()
    Y_fastmap = fastmap(X, distance, k)
    print("%s sec." % (time() - t0))
    print_evaluation(X, Y_fastmap)

    print("lMDS:")
    distance = euclidean_distance
    # distance = euclidean_distance_parallel
    t0 = time()
    Y_lmds = np.array(compute_lmds(X, nl=100, k=k, distance=distance,
                                   landmark_policy='sff'))
    print("%s sec." % (time() - t0))
    print_evaluation(X, Y_lmds)
