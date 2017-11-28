"""
__author__ = Yash Patel
__name__   = deanonymize.py
__description__ = Runs spectral clustering for deanonymization on the BTC network,
also calculating accuracy and drawing outputs in the process
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from constants import colors
from spectral import spectral_analysis, kmeans_analysis

def _reorder_clusters(clusters, partitions):
    reordered_partitions = [None for _ in clusters]
    used_partitions = set()
    for i, cluster in enumerate(clusters):
        intersects = np.array([len(cluster.intersection(partition)) 
             if j not in used_partitions else -1 for j, partition in enumerate(partitions)])
        most_similar = np.argmax(intersects)
        used_partitions.add(most_similar)
        reordered_partitions[i] = partitions[most_similar]
    return reordered_partitions

def draw_partitions(G, pos, clusters, partitions, fn):
    guessed_colors = [colors[j] for i in range(len(G.nodes))
        for j, partition in enumerate(partitions) if i in partition]

    nx.draw(G, pos, node_size=100, node_color=guessed_colors)
    plt.savefig("output/{}".format(fn))
    plt.close()

def calc_accuracy(truth, guess):
    if len(truth) == len(guess):
        guess = _reorder_clusters(truth, guess)

        num_correct = 0
        total_nodes = 0
        for i in range(len(truth)):
            num_correct += len(truth[i].intersection(guess[i]))
            total_nodes += len(truth[i])
        return 100.0 * (num_correct/total_nodes)
    return 0.0

def deanonymize(G, k):
    hier_partitions = spectral_analysis(G, k=k)
    kmeans_partitions = kmeans_analysis(G, k=k)
    return hier_partitions, kmeans_partitions