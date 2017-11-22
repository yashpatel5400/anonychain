"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Investigating eigenvalues of SBM model (toy examples)
"""
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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

    possible_colors = ["blue", "green", "red", "cyan", "black", "pink"]
    colors = np.random.choice(possible_colors, size=len(cluster_sizes), replace=False)
    graph_colors = [colors[i] for i in range(len(cluster_sizes)) 
        for _ in range(cluster_sizes[i])]

    spring_pos = nx.spring_layout(G)
    nx.draw(G, spring_pos, node_size=100, 
        alpha=0.75, node_color=graph_colors)
    plt.savefig("output/graph.png")
    plt.close()

    print("Prop same: {}; Prop diff: {}".format(
        verify_same/same_cluster_nodes, verify_diff/diff_cluster_nodes))
    return nx.to_numpy_matrix(G)