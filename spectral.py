"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Spectral analysis on the eigenvalues of the adjacency
matrix (for estimating number of distinct accounts)
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def _partition_graph(partition_eigenvector):
    partition1, partition2 = set(), set()
    for i in range(len(partition_eigenvector)):
        if partition_eigenvector[i] > 0:
            partition1.add(i)
        else: partition2.add(i)
    return [partition1, partition2]

def _spectral_partition(mat, eigen_index, mat_type):
    U, s, V = np.linalg.svd(mat)

    MARGIN_PROP = 0.75
    margin = MARGIN_PROP * np.std(s)
    e = [margin] * len(s)

    plt.errorbar(list(range(len(s))), s, yerr=e, fmt='o')
    plt.savefig("output/{}.png".format(mat_type))
    plt.close()

    partition_eigenvector = np.squeeze(np.asarray(U[eigen_index]))
    plt.scatter(list(range(len(partition_eigenvector))), 
        sorted(partition_eigenvector))
    plt.savefig("output/{}_eigenvector.png".format(mat_type))
    plt.close()

    return _partition_graph(partition_eigenvector)

def spectral_analysis(G, partitions):
    adj_partitions = _spectral_partition(nx.adjacency_matrix(G).todense(), 1,  "adjacency")
    lap_partitions = _spectral_partition(nx.laplacian_matrix(G).todense(), -2, "laplacian")
    return adj_partitions, lap_partitions