"""
__author__ = Yash Patel
__name__   = deanonymize.py
__description__ = Runs spectral clustering for deanonymization on the BTC network,
also calculating accuracy and drawing outputs in the process
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from analysis.constants import colors
from analysis.spectral import spectral_analysis, kmeans_analysis

try:
    import blocksci
except:
    pass

def convert_address_to_compact(address):
    return address.address_num + (int(address.type) << 29)

def convert_compact_to_address(chain, compact):
    address_type = compact >> 29
    address_num = compact - (address_type << 29)
    return chain.address_from_index(address_num, blocksci.address_type(address_type))

def write_results(partitions, index_to_id, fn):
    try:
        chain = blocksci.Blockchain("/blocksci/bitcoin")
    except:
        pass
        
    with open("output/{}".format(fn), "w") as f:
        for partition_id, partition in enumerate(partitions):
            try:
                node_addresses = [convert_compact_to_address(chain, 
                    index_to_id[node]) for node in partition]
            except:
                node_addresses = [index_to_id[node] for node in partition]
            f.writelines("{} : {}\n".format(partition_id, node_addresses))

def _reorder_clusters(clusters, partitions):
    """Given the ground truth clusters and partitions (list of sets, where the 
    contents of the first set are the nodes that belong to "cluster 1"), reorders the
    partitions to align with the ground truth clusters. This is to avoid the problem
    that calling a cluster "cluster 1" vs. calling it "cluster 2" is completely arbitrary,
    and so this shuffles the numbers to achieve the highest accuracy possible. Note that
    this should only be called in test settings (i.e. not for the final Bitcoin trials)

    Returns Partitions (list of sets of ints)
    """
    reordered_partitions = [None for _ in clusters]
    used_partitions = set()
    for i, cluster in enumerate(clusters):
        intersects = np.array([len(cluster.intersection(partition)) 
             if j not in used_partitions else -1 for j, partition in enumerate(partitions)])
        most_similar = np.argmax(intersects)
        used_partitions.add(most_similar)
        reordered_partitions[i] = partitions[most_similar]
    return reordered_partitions

def draw_results(G, pos, partitions, fn, weigh_edges=False):
    """Given a graph (G), the node positions (pos), the partitions on the nodes, the destination
    filename, and whether or not the edges are weighted, plots a figure and saves it
    to the destination location (in the output/ folder)

    Returns void
    """
    if len(partitions) > len(colors):
        print("Too many partitions to plot!")
        return

    print("Plotting graph partitions...")
    nodes = list(G.nodes)
    if partitions is None:
        guessed_colors = ["r"] * len(nodes)
    else:
        guessed_colors = [colors[j] for i in range(len(nodes))
            for j, partition in enumerate(partitions) if nodes[i] in partition]
    
    if weigh_edges:
        edgewidth = [d['weight'] for (u,v,d) in G.edges(data=True)]
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color=guessed_colors)
        nx.draw_networkx_edges(G, pos, width=edgewidth)
    else:
        nx.draw(G, pos, node_size=100, node_color=guessed_colors)

    plt.axis('off')
    plt.savefig("output/{}".format(fn))
    plt.close()

def calc_accuracy(truth, guess):
    """Given the ground truth and guessed partitions (both list of sets of ints), finds
    the accuracy of the clustering algorithm. Returns as a percent, i.e. between 0 and 100

    Returns Accuracy (float in [0.0,100.0])
    """
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
    """Given the input graph G and number of clusters k, runs hierarchical and k-means clustering
    to produce partitions. Returns these partitions as lists of sets of ints

    Returns (1) Partitions from hierarchical clustering
    (2) Partitions from k-means clustering
    """
    print("Running partitioning analyses on graph...")
    hier_partitions = spectral_analysis(G, k=k)
    kmeans_partitions = kmeans_analysis(G, k=k)
    return hier_partitions, kmeans_partitions