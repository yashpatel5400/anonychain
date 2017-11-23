"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Spectral analysis on the eigenvalues of the adjacency
matrix (for estimating number of distinct accounts)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def _eigenvectors(laplacian, fn):
    U, s, V = np.linalg.svd(laplacian)

    MARGIN_PROP = 0.75
    margin = MARGIN_PROP * np.std(s)
    e = [margin] * len(s)

    plt.errorbar(list(range(len(s))), s, yerr=e, fmt='o')
    plt.savefig("output/{}".format(fn))
    plt.close()

    return U

def _partition_graph(second_smallest_eigenvector):
    partition1, partition2 = set(), set()
    for i in range(len(second_smallest_eigenvector)):
        if second_smallest_eigenvector[i] > 0:
            partition1.add(i)
        else: partition2.add(i)
    return [partition1, partition2]

def spectral_analysis(G, partitions):
    laplacian = nx.laplacian_matrix(G).todense()
    U = _eigenvectors(laplacian, "laplacian.png")
    
    second_smallest_eigenvector = np.squeeze(np.asarray(U[-2]))
    plt.scatter(list(range(len(second_smallest_eigenvector))), 
        sorted(second_smallest_eigenvector))
    plt.savefig("output/second_smallest.png")
    plt.close()

    partitions = _partition_graph(second_smallest_eigenvector)
    return partitions