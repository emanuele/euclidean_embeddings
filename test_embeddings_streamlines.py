"""Simple tests for Euclidean embeddings of streamlines.
"""

import numpy as np
from dissimilarity import compute_dissimilarity
from fastmap import compute_fastmap
from lmds import compute_lmds
from scipy.spatial import distance_matrix
from evaluation_metrics import stress, correlation, distortion
from distances import euclidean_distance, parallel_distance_computation
from functools import partial
from time import time
import nibabel as nib
from test_embeddings import print_evaluation

def load(filename="data/sub-100307/sub-100307_var-FNAL_tract.trk"):
    print('Loading %s' % filename)
    data = nib.streamlines.load(filename)
    s = data.streamlines
    print("%s streamlines" % len(s))
    return s

if __name__ == '__main__':
    print(__doc__)
    np.random.seed(0)

    from dipy.tracking.distances import bundles_distances_mam
    X = np.array(load(), dtype=np.object)
    idx = np.random.permutation(X.shape[0])[:10000]
    X = X[idx]
    distance = bundles_distances_mam
    # distance = partial(parallel_distance_computation, distance=bundles_distances_mam)
    k = 20
    
    print("Estimating the time and quality of embedded distances vs. original distances.")

    print("Dissimilarity Representation:")
    t0 = time()
    Y_dissimilarity, prototype_idx = compute_dissimilarity(X, distance, k,
                                                           prototype_policy='sff',
                                                           verbose=False)
    print("%s sec." % (time() - t0))
    print_evaluation(X, distance, Y_dissimilarity)

    print("")
    print("Fastmap:")
    t0 = time()
    Y_fastmap = compute_fastmap(X, distance, k)
    print("%s sec." % (time() - t0))
    print_evaluation(X, distance, Y_fastmap)

    print("")
    print("lMDS:")
    t0 = time()
    Y_lmds = compute_lmds(X, distance, k, nl=40,
                          landmark_policy='sff')
    print("%s sec." % (time() - t0))
    print_evaluation(X, distance, Y_lmds)
