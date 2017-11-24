"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Spectral analysis on the eigenvalues of the adjacency
matrix (for estimating number of distinct accounts)
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans

def _partition_graph(G, partition_eigenvector):
    partition1, partition2 = set(), set()
    for i in range(len(partition_eigenvector)):
        if partition_eigenvector[i] > 0:
            partition1.add(i)
        else: partition2.add(i)
    return [G.subgraph(partition1), G.subgraph(partition2)]

def _plot_eigenvalues(s, fn):
    MARGIN_PROP = 0.75
    margin = MARGIN_PROP * np.std(s)
    e = [margin] * len(s)

    plt.errorbar(list(range(len(s))), s, yerr=e, fmt='o')
    plt.savefig("output/{}".format(fn))
    plt.close()

def _plot_eigenvector(eigenvector, fn):
    plt.scatter(list(range(len(eigenvector))), 
        sorted(eigenvector))
    plt.savefig("output/{}".format(fn))
    plt.close()

def _spectral_partition(G, mat_type):
    EIGEN_GAP   = .50
    eigen_index = 1

    if mat_type == "adjacency":
        get_mat = lambda G : nx.adjacency_matrix(G).todense()
        LOWER_BOUND = 2.0
    else:
        get_mat = lambda G : nx.laplacian_matrix(G).todense()
        LOWER_BOUND = 2.0
    
    k = None
    partitions = [G]
    while True:
        largest_partition = np.argmax(np.array([len(graph.nodes) for graph in partitions]))
        to_partition = partitions[largest_partition]
        mat = get_mat(to_partition)

        U, s, _ = np.linalg.svd(mat)
        eigenvectors = U # np.transpose(U)
        if mat_type == "laplacian":
            eigenvectors = eigenvectors[::-1]
            s = s[::-1]

        partition_eigenvalue  = s[1]
        partition_eigenvector = np.squeeze(np.asarray(eigenvectors[eigen_index]))

        if k is None:
            eigen_steps = [(s[i] - s[i-1]) / s[i-1] for i in range(1, len(s))] 
            for i, eigen_step in enumerate(eigen_steps):
                k = i
                if eigen_step > EIGEN_GAP:
                    break
        
        if len(partitions) >= k:
            break

        _plot_eigenvalues(s, "{}/eigenvalues_{}.png".format(mat_type, len(partitions)))
        _plot_eigenvalues(partition_eigenvector, 
            "{}/eigenvector_{}.png".format(mat_type, len(partitions)))

        new_partitions = _partition_graph(to_partition, partition_eigenvector)
        del partitions[largest_partition] 
        partitions += new_partitions
    print("Completed {} partitioning w/ {} partitions".format(mat_type, k))
    return partitions

def spectral_analysis(G, partitions):
    adj_partitions = _spectral_partition(G, "adjacency")
    lap_partitions = _spectral_partition(G, "laplacian")
    return adj_partitions, lap_partitions

def kmeans_analysis(G, clusters, k):
    L = nx.laplacian_matrix(G).todense()
    U, _, _ = np.linalg.svd(L)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(U)

    true_labels = np.array([j for i in range(len(kmeans.labels_))
            for j, cluster in enumerate(clusters) if i in cluster])
    accuracy = sum(true_labels == kmeans.labels_) / len(true_labels)
    print("K-means accuracy: {}".format(accuracy))