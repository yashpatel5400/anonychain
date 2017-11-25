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
    EIGEN_GAP   = 1.75
    
    if mat_type == "adjacency":
        get_mat = lambda G : nx.adjacency_matrix(G).todense()
        eigen_index = 1
    else:
        get_mat = lambda G : nx.laplacian_matrix(G).todense()
        eigen_index = -2

    k = 2
    partitions = [G]
    while True:
        largest_partition = np.argmax(np.array([len(graph.nodes) for graph in partitions]))
        to_partition = partitions[largest_partition]
        mat = get_mat(to_partition)

        U, s, _ = np.linalg.svd(mat)
        eigenvectors = np.transpose(U)
        
        partition_eigenvalue  = s[eigen_index]
        partition_eigenvector = np.squeeze(np.asarray(eigenvectors[eigen_index]))

        if k is None:
            rev_s = s[::-1]
            eigen_steps = [(rev_s[i] - rev_s[i-1]) for i in range(1, len(rev_s))] 
            print(eigen_steps)
            for i, eigen_step in enumerate(eigen_steps):
                k = i + 1
                if eigen_step > EIGEN_GAP:
                    break
            print("Partitioning into {} clusters".format(k))

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
    eigenvectors = np.transpose(U)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigenvectors)

    true_labels = np.array([j for i in range(len(kmeans.labels_))
            for j, cluster in enumerate(clusters) if i in cluster])
    accuracy = sum(true_labels == kmeans.labels_) / len(true_labels)
    print("K-means accuracy: {}".format(accuracy))