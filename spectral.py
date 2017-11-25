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
    partition1, partition2, partition3 = set(), set(), set()
    for i in range(len(partition_eigenvector)):
        cur_node = list(G.nodes())[i]
        if partition_eigenvector[i] > 0:
            partition1.add(cur_node)
        elif partition_eigenvector[i] == 0:
            partition2.add(cur_node)
        else: partition3.add(cur_node)

    partitions = [partition1, partition2, partition3]
    nonempty_partitions = [partition for partition in 
        partitions if len(partition) != 0]
    return [G.subgraph(partition) for partition in nonempty_partitions]

def _plot_eigenvalues(s, fn):
    MARGIN_PROP = 0.75
    margin = MARGIN_PROP * np.std(s)
    e = [margin] * len(s)

    plt.errorbar(list(range(len(s))), s, yerr=e, fmt='o')
    # plt.yscale('log')
    plt.savefig("output/{}".format(fn))
    plt.close()

def _plot_eigenvector(eigenvector, fn):
    plt.scatter(list(range(len(eigenvector))), 
        sorted(eigenvector))
    plt.savefig("output/{}".format(fn))
    plt.close()

def _spectral_partition(G, mat_type, normalize=True):
    EIGEN_GAP = 1.0
    
    if mat_type == "adjacency":
        get_mat = lambda G : nx.adjacency_matrix(G).todense()
        eigen_index = 1
    else:
        if normalize:
            get_mat = lambda G : nx.normalized_laplacian_matrix(G).todense()
        else: get_mat = lambda G : nx.laplacian_matrix(G).todense()
        eigen_index = -2

    k = 3
    partitions = [G]
    while True:
        second_least_eigenvalues = []
        for partition in partitions:
            _, s, _ = np.linalg.svd(get_mat(partition))
            second_least_eigenvalues.append(s[-2])

        # largest_partition = np.argmax(np.array([len(graph.nodes) for graph in partitions]))
        best_partition = np.argmin(np.array(second_least_eigenvalues))
        to_partition = partitions[best_partition]
        mat = get_mat(to_partition)

        U, s, _ = np.linalg.svd(mat)
        eigenvectors = np.transpose(U)
        
        partition_eigenvalue  = s[eigen_index]
        partition_eigenvector = np.squeeze(np.asarray(eigenvectors[eigen_index]))

        _plot_eigenvalues(s, "{}/eigenvalues_{}.png".format(mat_type, len(partitions)))
        _plot_eigenvector(partition_eigenvector, 
            "{}/eigenvector_{}.png".format(mat_type, len(partitions)))

        if k is None:
            smallest_eigenvalues = np.array(s[::-1][:10])
            _plot_eigenvalues(smallest_eigenvalues, "smallest_eigenvalues.png")
            eigen_steps = [(smallest_eigenvalues[i] - smallest_eigenvalues[i-1]) 
                for i in range(1, len(smallest_eigenvalues))] 
            for i, eigen_step in enumerate(eigen_steps):
                if eigen_step > EIGEN_GAP:
                    k = i + 1
                    break
            print("Partitioning into {} clusters".format(k))

        if len(partitions) >= k:
            break

        new_partitions = _partition_graph(to_partition, partition_eigenvector)
        del partitions[best_partition] 
        partitions += new_partitions

        print([partition.nodes() for partition in partitions])
    print("Completed {} partitioning w/ {} partitions".format(mat_type, k))
    return partitions

def spectral_analysis(G, partitions):
    # adj_partitions = _spectral_partition(G, "adjacency")
    lap_partitions = _spectral_partition(G, "laplacian", normalize=True)
    return lap_partitions #, adj_partitions

def kmeans_analysis(G, clusters, k):
    L = nx.laplacian_matrix(G).todense()
    U, _, _ = np.linalg.svd(L)
    eigenvectors = np.transpose(U)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigenvectors)

    true_labels = np.array([j for i in range(len(kmeans.labels_))
            for j, cluster in enumerate(clusters) if i in cluster])
    accuracy = sum(true_labels == kmeans.labels_) / len(true_labels)
    print("K-means accuracy: {}".format(accuracy))