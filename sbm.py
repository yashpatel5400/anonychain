"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Investigating eigenvalues of SBM model (toy examples)
"""
import random
import networkx as nx
import numpy as np

def _create_clusters(cluster_sizes):
    completed_nodes = 0
    clusters = []
    for cluster_size in cluster_sizes:
        clusters.append(set(range(completed_nodes, completed_nodes + cluster_size)))
        completed_nodes += cluster_size
    return clusters

def create_sbm(cluster_sizes, p, q):
    """
    creates a stochastic block model, with probability p within the same
    cluster and probability q outside, with n vertices
    """
    G = nx.Graph()
    total_nodes = sum(cluster_sizes)
    G.add_nodes_from(list(range(total_nodes)))

    clusters = _create_clusters(cluster_sizes)
    verify_same = 0
    verify_diff = 0

    same_cluster_nodes = 0
    diff_cluster_nodes = 0
    for cluster in clusters:
        for cur_node in cluster:
            for other_node in range(total_nodes):
                if other_node == cur_node:
                    continue
                if other_node in cluster:
                    prob = p
                    same_cluster_nodes += 1
                else: 
                    prob = q
                    diff_cluster_nodes += 1                    

                if random.random() < prob:
                    if other_node in cluster:
                        verify_same += 1
                    else: verify_diff += 1
                    G.add_edge(cur_node, other_node)

    print("Prop same: {}; Prop diff: {}".format(
        verify_same/same_cluster_nodes, verify_diff/diff_cluster_nodes))
    return nx.to_numpy_matrix(G)

def get_eigenvectors(A):
    MARGIN = .25
    U, s, V = np.linalg.svd(A)

    rep_eigenvectors = U[:2]
    min_distances = []
    for eigenvector in U:
        distances = [np.linalg.norm(eigenvector - rep_eigenvector) for 
            rep_eigenvector in rep_eigenvectors]
        min_distances.append(min(distances))

    distinct_jumps = []
    for i in range(1, len(s)):
        if (s[i-1] - s[i])/s[i-1] > MARGIN:
            distinct_jumps.append(s[i-1])
    # print(distinct_jumps)