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
from sklearn.cluster import KMeans, SpectralClustering
from scipy.sparse.linalg import svds

def _partition_graph(G, partition_eigenvector):
    """Given a graph G and the eigenvector to be used for partitioning, separates
    the nodes into three separate sets, with the first as those whose components
    are strictly positive, second equal to 0, and third strictly negative

    Returns Partitions (list of NetworkX Subgraphs)
    """
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

def _plot_eigenvalues(eigenvalues, fn):
    """Given a list of eigenvalues and filename, plots the eigenvalues 
    and saves at filename

    Returns void
    """
    MARGIN_PROP = 0.75
    margin = MARGIN_PROP * np.std(eigenvalues)
    e = [margin] * len(eigenvalues)

    plt.errorbar(list(range(len(eigenvalues))), eigenvalues, yerr=e, fmt='o')
    plt.axhline(0.1)
    plt.savefig("output/{}".format(fn))
    plt.close()

def _plot_eigenvector(eigenvector, fn):
    """Given an eigenvector and filename, plots the eigenvector and saves at filename

    Returns void
    """
    plt.scatter(list(range(len(eigenvector))), 
        sorted(eigenvector))
    plt.savefig("output/{}".format(fn))
    plt.close()

def spectral_analysis_alt(L, k=None, normalize=True):
    """Given an input graph (G), number of clusters (k), and whether the graph
    Laplacian is to be normalized (True) or not (False) runs spectral clustering
    as implemented in scikit-learn (empirically found to be less effective)

    Returns Partitions (list of sets of ints)
    """
    sc = SpectralClustering(k, n_init=10, affinity="precomputed")
    labels = sc.fit_predict(L)
    
    partitions = [set() for _ in range(k)]
    for i, guess in enumerate(labels):
        partitions[guess].add(i)
    return partitions

def spectral_analysis(G, k=None, normalize=True):
    """Given an input graph (G), number of clusters (k), and whether the graph
    Laplacian is to be normalized (True) or not (False) runs spectral clustering
    on the graph Laplacian using hierarchial method. Clusters are returned as a list of sets,
    where the contents of the first set are the nodes that belong to "cluster 1"

    Returns Partitions (list of sets of ints)
    """
    EIGEN_GAP = 0.1
    
    if normalize:
        # get_mat = lambda G : nx.normalized_laplacian_matrix(G).todense()
        get_mat = lambda G : nx.normalized_laplacian_matrix(G).asfptype()
    else: 
        # get_mat = lambda G : nx.laplacian_matrix(G).todense()
        get_mat = lambda G : nx.laplacian_matrix(G).asfptype()
    
    partitions = [G]
    while True:
        second_least_eigenvalues = []

        min_partition_eigenvalue = None
        best_partition = None
        partition_eigenvector = None

        for i, partition in enumerate(partitions):
            if len(partition.nodes) > 1:
                mat = get_mat(partition)
                
                # in the case of having 2 nodes, the 2nd least eigenvalue is the largest eigenvalue
                if len(partition.nodes) == 2:
                    U, s, _ = svds(mat, k=1, which='LM', return_singular_vectors="u")
                    cur_eigenvector = U[:, 0]
                    partition_eigenvalue = s[0]
                
                # else we can just use the smallest two eigenvalues
                else:
                    U, s, _ = svds(mat, k=2, which='SM', return_singular_vectors="u")
                    cur_eigenvector = U[:, 1]
                    partition_eigenvalue = s[1]
                
                # _, s, _ = np.linalg.svd(get_mat(partition))
                if min_partition_eigenvalue is None or partition_eigenvalue < min_partition_eigenvalue:
                    best_partition = i
                    partition_eigenvector = cur_eigenvector
                    min_partition_eigenvalue = partition_eigenvalue

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
                if eigen_step > EIGEN_GAP:
                    k = i + 1
            if k is None:
                k = 1
            print("Partitioning into {} clusters".format(k))

        if len(partitions) >= k:
            break

        print(best_partition)

        new_partitions = _partition_graph(partitions[best_partition], partition_eigenvector)
        del partitions[best_partition] 
        
        if len(partitions + new_partitions) > k:
            new_partitions = [nx.compose(new_partitions[0], new_partitions[1]), new_partitions[2]]
        partitions += new_partitions

        print([partition.nodes() for partition in partitions])
    print("Completed partitioning w/ {} partitions".format(k))
    # return partitions

    partitions = [set(partition.nodes()) for partition in partitions]
    return partitions

def kmeans_analysis(G, k):
    """Given an input graph (G) and number of clusters (k), runs spectral clustering
    on the graph Laplacian using k-means. Clusters are returned as a list of sets,
    where the contents of the first set are the nodes that belong to "cluster 1"

    Returns Partitions (list of sets of ints)
    """
    print("Partitioning w/ k-means on {} clusters".format(k))
    
    L = nx.laplacian_matrix(G).asfptype()
    return kmean_spectral(L)

def kmean_spectral(L, k):
    """Given an input matrix and number of clusters k, runs spectral clustering
    on the graph Laplacian using k-means. Clusters are returned as a list of sets,
    where the contents of the first set are the nodes that belong to "cluster 1"

    Returns Partitions (list of sets of ints)
    """
    U, _, _ = svds(L, k=k, which='SM', return_singular_vectors="u")

    guesses = KMeans(n_clusters=k, n_jobs=-1).fit_predict(U)
    partitions = [set() for _ in range(k)]
    for i, guess in enumerate(guesses):
        partitions[guess].add(i)
    print("Completed k-means partitioning")
    return partitions