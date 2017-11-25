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
    plt.savefig("output/{}".format(fn))
    plt.close()

def _plot_eigenvector(eigenvector, fn):
    plt.scatter(list(range(len(eigenvector))), 
        sorted(eigenvector))
    plt.savefig("output/{}".format(fn))
    plt.close()

def spectral_analysis(G, normalize=True):
    EIGEN_GAP = 1.0
    
    if normalize:
        get_mat = lambda G : nx.normalized_laplacian_matrix(G).todense()
    else: get_mat = lambda G : nx.laplacian_matrix(G).todense()
    
    k = 4
    partitions = [G]
    while True:
        second_least_eigenvalues = []
        for partition in partitions:
            _, s, _ = np.linalg.svd(get_mat(partition))
            second_least_eigenvalues.append(s[-2])

        best_partition = np.argmin(np.array(second_least_eigenvalues))
        to_partition = partitions[best_partition]
        mat = get_mat(to_partition)

        U, s, _ = np.linalg.svd(mat)
        eigenvectors = np.transpose(U)
        
        partition_eigenvalue  = s[-2]
        partition_eigenvector = np.squeeze(np.asarray(eigenvectors[-2]))

        _plot_eigenvalues(s, "eigen/eigenvalues_{}.png".format(len(partitions)))
        _plot_eigenvector(partition_eigenvector, 
            "eigen/eigenvector_{}.png".format(len(partitions)))

        if k is None:
            smallest_eigenvalues = np.array(s[::-1][:10])
            eigen_steps = [(smallest_eigenvalues[i] - smallest_eigenvalues[i-1]) 
                for i in range(1, len(smallest_eigenvalues))] 
            _plot_eigenvalues(smallest_eigenvalues, "smallest_eigenvalues.png")
            _plot_eigenvalues(eigen_steps, "eigen_step.png")

            for i, eigen_step in enumerate(eigen_steps):
                k = i + 1
                if eigen_step > EIGEN_GAP:
                    break
            print("Partitioning into {} clusters".format(k))

        if len(partitions) >= k:
            break

        new_partitions = _partition_graph(to_partition, partition_eigenvector)
        del partitions[best_partition] 
        
        if len(partitions + new_partitions) > k:
            new_partitions = [new_partitions[0] + new_partitions[1], new_partitions[2]]
        partitions += new_partitions

        print([partition.nodes() for partition in partitions])
    print("Completed partitioning w/ {} partitions".format(k))
    return partitions

def kmeans_analysis(G, clusters, k):
    L = nx.laplacian_matrix(G).todense()
    U, _, _ = np.linalg.svd(L)
    eigenvectors = np.transpose(U)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigenvectors)

    true_labels = np.array([j for i in range(len(kmeans.labels_))
            for j, cluster in enumerate(clusters) if i in cluster])
    accuracy = sum(true_labels == kmeans.labels_) / len(true_labels)
    print("K-means accuracy: {}".format(accuracy))