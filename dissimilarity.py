"""Computation of the dissimilarity representation of a set of objects
from a set of prototypes given a distance function. Some prototype
selection algorithms are available.

See Olivetti E., Nguyen T.B., Garyfallidis, E., The Approximation of
the Dissimilarity Projection, http://dx.doi.org/10.1109/PRNI.2012.13

"""

from __future__ import division
import numpy as np
from subsampling import compute_subset
try:
    from joblib import Parallel, delayed, cpu_count
    joblib_available = True
except:
    joblib_available = False


def compute_dissimilarity(tracks, num_prototypes=40, distance=None,
                          prototype_policy='sff', n_jobs=-1,
                          verbose=False):
    """Compute the dissimilarity (distance) matrix between tracks and
    prototypes, where prototypes are selected among the tracks with a
    given policy.

    Parameters
    ----------
    tracks : list or array of objects
           an iterable of streamlines.
    num_prototypes : int
           The number of prototypes. In most cases 40 is enough, which
           is the default value.
    distance : function
           Distance function between groups of streamlines. The
           default is bundles_distances_mam
    prototype_policy : string
           Shortname for the prototype selection policy. The default
           value is 'sff'.
    n_jobs : int
           If joblib is available, split the dissimilarity computation
           in n_jobs. If n_jobs is -1, then all available cpus/cores
           are used. The default value is -1.
    verbose : bool
           If true prints some messages. Deafault is True.

    Return
    ------
    dissimilarity_matrix : array (N, num_prototypes)

    See Also
    --------
    furthest_first_traversal, subset_furthest_first

    Notes
    -----
    """
    if verbose:
        print("Generating %s prototypes with policy %s." % (num_prototypes, prototype_policy))

    prototype_idx = compute_subset(tracks, num_prototypes, distance,
                                   landmark_policy=prototype_policy)
    prototypes = [tracks[i] for i in prototype_idx]
    dissimilarity_matrix = distance(tracks, prototypes, n_jobs=n_jobs,
                                    verbose=verbose)
    return dissimilarity_matrix, prototype_idx
