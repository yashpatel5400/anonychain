"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Investigating eigenvalues of SBM model (toy examples)
"""

import random
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_clusters(cluster_sizes):
    completed_nodes = 0
    clusters = []
    for cluster_size in cluster_sizes:
        clusters.append(set(range(completed_nodes, completed_nodes + cluster_size)))
        completed_nodes += cluster_size
    return clusters

def create_sbm(clusters, p, q, is_weighted):
    """
    creates a stochastic block model, with probability p within the same
    cluster and probability q outside, with n vertices
    """
    G = nx.Graph()
    total_nodes = sum([len(cluster) for cluster in clusters])
    G.add_nodes_from(list(range(total_nodes)))

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

                join_prob = random.random()
                if join_prob < prob:
                    if other_node in cluster:
                        verify_same += 1
                    else: verify_diff += 1

                    if is_weighted:
                        random_weight = 1 - join_prob
                        G.add_edge(cur_node, other_node, weight=random_weight)
                    else:
                        G.add_edge(cur_node, other_node)

    print("Prop same: {}; Prop diff: {}".format(
        verify_same/same_cluster_nodes, verify_diff/diff_cluster_nodes))
    return G