"""Metrics to evaluate the quality of an embedding.
"""

import numpy as np
from scipy.stats.stats import pearsonr


def stress(dist_embedd, dist_original):
    tmp = dist_embedd / (dist_embedd * dist_embedd).sum() - \
          dist_original / (dist_original * dist_original).sum()
    return (tmp * tmp).sum()


def correlation(dist_embedd, dist_original):
    return np.corrcoef(dist_embedd, dist_original)[1,0]


def distortion(dist_embedd, dist_original, eps=1.0e-20):
    """Distortion (c1*c2) of the embedded distances with respect to the
    original distances, where
       (1/c1)*do(o1, o2) <= de(f(o1), f(o2)) <= c2*do(o1,o2)
    for all pairs of objects and c1, c2 >= 1.

    See: Samet, "Foundations of Multidimensional and Metric Data
    Structures", Sec.4.7.1.
    """
    c1 = np.max([(dist_original / (dist_embedd + eps)).max(), 1.0])
    c2 = np.max([(dist_embedd / (dist_original + eps)).max(), 1.0])
    return c1 * c2


if __name__ == '__main__':
    dist_original = np.arange(1, 20)
    dist_embedd = dist_original ** 2 + np.random.normal(len(dist_original)) * 0.5
    print("stress = %s" % stress(dist_embedd, dist_original))
    print("correlation = %s" % correlation_distance(dist_embedd, dist_original))
    print("distortion = %s" % distortion(dist_embedd, dist_original))
    
