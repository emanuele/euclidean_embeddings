"""Lipschitz embedding.

See: Samet, "Foundations of Multidimensional and Metric Data
Structures", Sec.4.7.2.
"""

import numpy as np
from joblib import Parallel, delayed
import multiprocessing


def compute_reference_sets(dataset, k, sizeA=None):
    """Compute k reference sets for a given dataset. Optionally, specify
    the size of the reference sets.
    """
    sizeD = len(dataset)
    if sizeA is None:
        sizeA = [(np.random.randint(sizeD) + 1) for i in range(k)]
        
    R = []
    for i in range(k):
        A_i = dataset[np.random.permutation(sizeD)[: sizeA[i]]]
        R.append(A_i)

    return R


def compute_reference_sets_linial1994(dataset, k=None):
    """Compute the reference set according to the theorem of Linial et. al
    "The geometry of graphs and some of its algorithmic applications", 1994.

    """
    sizeD = len(dataset)
    R = []
    for i in range(k):
        size_i = 2 ** int(np.floor(float(i) / np.log2(sizeD) + 1.0))
        A_i = dataset[np.random.permutation(sizeD)[:size_i]]
        R.append(A_i)

    return R


def compute_distance_from_reference_set(object, A, distance_function):
    return np.min([distance_function([object], [x]) for x in A])


def compute_distance_from_reference_sets(object, R, distance_function):
    return np.array([compute_distance_from_reference_set(object,
                                                         A,
                                                         distance_function) for A in R])


def compute_distance_from_reference_set_fast(dataset, A,
                                             distance_function):
    return distance_function(dataset, A).min(1)


def compute_distance_from_reference_sets_fast(dataset, R,
                                              distance_function):
    dataset_embedded = np.zeros((len(dataset), len(R)),
                                dtype=np.float)
    for i, A in enumerate(R):
        dataset_embedded[:, i] = compute_distance_from_reference_set_fast(dataset,
                                                                          A,
                                                                          distance_function)

    return dataset_embedded
    

def compute_lipschitz(dataset, distance_function, k=None,
                      linial1994=True, sizeA=None, p=2, k_max=None,
                      linial1994_normalize=False):
    """Compute the Lipschitz embedding of a given dataset of objects,
    given its distance_function.

    Optional parameters: the target dimension k and whether to choose
    the sizes of the reference sets following the theorem of Linial et
    al. (1994), in order to have explicit bounds on the embedding.

    """
    if k is None:
        k = int(np.floor(np.log2(len(dataset)))) ** 2
        print("k = %s" % k)

    if k_max is not None and k > k_max:
        k = k_max
        print("k = %s" % k)

    if linial1994:
        R = compute_reference_sets_linial1994(dataset, k)
    else:
        R = compute_reference_sets(dataset, k, sizeA)

    # Slow:
    # dataset_embedded = np.zeros([len(dataset), k])
    # for i, object in enumerate(dataset):
    #     print(i)
    #     dataset_embedded[i, :] = compute_distance_from_reference_sets(object,
    #                                                                   R,
    #                                                                   distance_function)

    # Fast:
    dataset_embedded = compute_distance_from_reference_sets_fast(dataset,
                                                                 R,
                                                                 distance_function)

    if linial1994_normalize:
        dataset_embedded = dataset_embedded / (k ** (1.0 / p))
        
    return dataset_embedded, R
