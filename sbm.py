"""
__author__ = Yash Patel
__name__   = sbm.py
__description__ = Investigating eigenvalues of SBM model (toy examples)
"""

import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_sbm(clusters, p, q):
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

                if random.random() < prob:
                    if other_node in cluster:
                        verify_same += 1
                    else: verify_diff += 1
                    G.add_edge(cur_node, other_node)

    print("Prop same: {}; Prop diff: {}".format(
        verify_same/same_cluster_nodes, verify_diff/diff_cluster_nodes))
    return G