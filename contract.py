"""
__author__ = Yash Patel
__name__   = contract.py
__description__ = Main contractions script
"""

import networkx as nx
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from setup.sbm import create_sbm, create_clusters
from analysis.pca import plot_pca
from analysis.deanonymize import draw_partitions, calc_accuracy, deanonymize
from blockchain.read import create_simple_graph

def _contract_edges(G, num_edges):
    identified_nodes = {}
    for _ in range(num_edges):
        edges = list(G.edges)
        random_edge = edges[random.randint(0,len(edges)-1)]
        node_to_contract = random_edge[1]
        identified_nodes[node_to_contract] = random_edge[0] # right gets contracted into left
        G = nx.contracted_edge(G, random_edge)
    return G, identified_nodes

def _reconstruct_contracted(identified_nodes, partitions):
    for contracted in identified_nodes:
        for partition in partitions:
            if identified_nodes[contracted] in partition:
                partition.add(contracted)
                break
    return partitions

def contract_deanonymize(G, k):
    contracted_G, identified_nodes = _contract_edges(G, num_edges=10)
    hier_partitions, kmeans_partitions = deanonymize(contracted_G, k=k)
    hier_partitions = _reconstruct_contracted(identified_nodes, hier_partitions)
    kmeans_partitions = _reconstruct_contracted(identified_nodes, kmeans_partitions)
    return hier_partitions, kmeans_partitions

def test_contract():
    cluster_size = 8
    num_clusters = 5
    cluster_sizes = [cluster_size] * num_clusters
    clusters = create_clusters(cluster_sizes)

    G = create_sbm(clusters, 1.0, 0.0, False)
    num_clusters = len(clusters)
    hier_partitions, kmeans_partitions = contract_deanonymize(G, k=num_clusters)
    
    print("hierarchical accuracy: {}".format(calc_accuracy(clusters, hier_partitions)))
    print("k-means accuracy: {}".format(calc_accuracy(clusters, kmeans_partitions)))
    
    spring_pos = nx.spring_layout(G)
    draw_partitions(G, spring_pos, clusters, 
        "contraction/truth.png", weigh_edges=False)
    draw_partitions(G, spring_pos, hier_partitions, 
        "contraction/eigen_guess.png", weigh_edges=False)
    draw_partitions(G, spring_pos, kmeans_partitions, 
        "contraction/kmeans_guess.png", weigh_edges=False)

if __name__ == "__main__":
    test_contract()