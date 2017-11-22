"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Spectral analysis on the eigenvalues of the adjacency
matrix (for estimating number of distinct accounts)
"""

import numpy as np
import matplotlib.pyplot as plt

def get_eigenvectors(A):
    U, s, V = np.linalg.svd(A)

    MARGIN_PROP = 0.75
    LOWER_BOUND = .25

    lower_bound = LOWER_BOUND * s[0]
    margin = MARGIN_PROP * np.std(s)
    e = [margin] * len(s)

    plt.errorbar(list(range(len(s))), s, yerr=e, fmt='o')
    plt.axhline(lower_bound)
    plt.savefig("output/eigenvalues.png")

    #rep_eigenvectors = U
    #min_distances = []
    #for eigenvector in U:
    #    distances = [np.linalg.norm(eigenvector - rep_eigenvector) for 
    #        rep_eigenvector in rep_eigenvectors]
    #    min_distances.append(min(distances))

    relevant_eigenvalues = [eigenvalue for eigenvalue in s if eigenvalue > lower_bound]
    print(relevant_eigenvalues)

    #distinct_jumps = []
    #for i in range(1, len(s)):
    #    if (s[i-1] - s[i])/s[i-1] > MARGIN:
    #        distinct_jumps.append(s[i-1])
    #print(distinct_jumps)