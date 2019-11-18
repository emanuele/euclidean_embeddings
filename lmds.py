"""
LMDS algorithm.
references:
      -https://www.sciencedirect.com/science/article/pii/S0031320308005049#sec2
      -https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7214117
      -http://www.cmap.polytechnique.fr/~peyre/cours/x2005signal/dimreduc_landmarks.pdf

1. choose n landmark points
2. compute distance-matrix Dl of landmax points, using the original distance
3. apply MDS to landmark points to obtain Euclidean embedding of landmarks of size k
4. compute U: the eigenvectors matrix and E: the diagonal eigenvalue matrix
5. compute E ^ (-1/2) * transpose (U)
6. compute median column means of Dl: mu
7. calculate squared distances of each point from n landmark points d_i
8.1 project points on eigenvectors to obtain embedding coordinates, use formula:
       y_i = 0.5 * pseudoinv_transpose(M) * ( mu - d_i )
where pseudoinv_transpose(M) is defined as : E ^ (-1/2) * transpose (U)
8.2 possible alternative: use only a subset of eigenvectors having the biggest
eigenvalues, dimention of embedded space in this case is n' < n
"""

import numpy as np
from .subsampling import compute_subset


def compute_lmds(dataset, distance, k, nl=100,
                 landmark_policy='random',eps=1.0e-10):
    """Given a dataset, computes the lMDS Euclidean embedding of size k
    using nl landmarks. The dataset must allow advanced indexing. In
    some cases, less than k dimensions ay be returned, specifically
    when less than k non negative eigenvalues of Dl are available.
    """
    landmarks_idx = compute_subset(dataset, distance,
                                   num_landmarks=nl,
                                   landmark_policy=landmark_policy)

    d = distance(dataset, dataset[landmarks_idx])

    # Use squared distances
    d *= d
    D = distance(dataset[landmarks_idx], dataset[landmarks_idx])
    D *= D

    # Compute cMDS on landmarks and get top k eigenvalues/eigenvectors
    n = D.shape[0]
    H = np.identity(n) - (1.0 / n) * np.ones((n, n))
    B = -0.5 * H.dot(D).dot(H)
    Lambda, U = np.linalg.eigh(B)  # eigh() because B (and D) must be symmetric
    U = U.T  # U has eigenvectors on columns. We prefer on rows.
    idx = Lambda.argsort()[::-1]
    k_max = (Lambda > eps).sum()
    if k > k_max:
        k = k_max
        print("WARNING: I cannot obtain more than %s dimensions." % k)

    Lambda_plus = Lambda[idx][:k]
    U_plus = U[idx][:k]

    # Compute M_sharp and the embedding:
    M_sharp = np.diag(1.0 / np.sqrt(Lambda_plus)).dot(U_plus)
    Y = (0.5 * M_sharp.dot((D.mean(0) - d).T)).T

    return Y
