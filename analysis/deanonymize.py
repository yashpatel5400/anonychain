"""
__author__ = Yash Patel
__name__   = deanonymize.py
__description__ = Runs spectral clustering for deanonymization on the BTC network,
also calculating accuracy and drawing outputs in the process
"""

import pickle
import time
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

    partition_to_nodes = {}
    with open("output/{}.txt".format(fn), "w") as f:
        for partition_id, partition in enumerate(partitions):
            try:
                node_addresses = { convert_compact_to_address(chain, 
                    index_to_id[node]).address_string for node in partition }
            except:
                node_addresses = { index_to_id[node] for node in partition }
            f.writelines("{} : {}\n".format(partition_id, node_addresses))
            partition_to_nodes[partition_id] = node_addresses
    pickle.dump(partition_to_nodes, open("output/{}.pickle".format(fn),"wb"))

def draw_results(G, pos, partitions, fn, weigh_edges=False, outliers=None):
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
        guessed_colors = []
        for i in range(len(nodes)):
            if outliers is not None and nodes[i] in outliers:
                guessed_colors.append("k")
            for j, partition in enumerate(partitions):
                if nodes[i] in partition:
                    guessed_colors.append(colors[j])
    
    if weigh_edges:
        edgewidth = [d['weight'] for (u,v,d) in G.edges(data=True)]
        nx.draw_networkx_nodes(G, pos, node_size=100, node_color=guessed_colors)
        nx.draw_networkx_edges(G, pos, width=edgewidth)
    else:
        nx.draw(G, pos, node_size=100, node_color=guessed_colors)

    plt.axis('off')
    plt.savefig("output/{}".format(fn))
    plt.close()

def _reorder_clusters(clusters, partitions, intersect_allowed=True):
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
        if intersect_allowed:
            intersects = np.array([len(cluster.intersection(partition)) for 
                partition in partitions])
        
        else:
            intersects = np.array([len(cluster.intersection(partition)) 
                if j not in used_partitions else -1 for j, partition in enumerate(partitions)])
        
        most_similar = np.argmax(intersects)
        used_partitions.add(most_similar)
        reordered_partitions[i] = partitions[most_similar]
    return reordered_partitions

def calc_accuracy(truth, guess, n):
    """Given the ground truth and guessed partitions (both list of sets of ints), finds
    the accuracy of the clustering algorithm. Returns as a percent, i.e. between 0 and 100

    Returns Accuracy (float in [0.0,100.0])
    """
    if len(truth) == len(guess):
        guess = _reorder_clusters(truth, guess, intersect_allowed=False)

        num_correct = 0
        for i in range(len(truth)):
            num_correct += len(truth[i].intersection(guess[i]))
        return 100.0 * (num_correct/n)
    return 0.0

# accuracy measures below according to: 
# https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html

def calc_accuracies(truth, guess, n):
    return {
        "purity"            : calc_purity  (truth, guess, n),
        "nmi"               : calc_nmi     (truth, guess, n),
        "rand_ind"          : calc_rand_ind(truth, guess, n),
        "weighted_ri"       : calc_rand_ind(truth, guess, n, beta=0.5),   
    }

def calc_purity(truth, guess, n):
    if len(truth) == len(guess):
        guess = _reorder_clusters(truth, guess, intersect_allowed=True)

        num_correct = 0
        for i in range(len(truth)):
            num_correct += len(truth[i].intersection(guess[i]))
        return num_correct/n
    return 0.0

def calc_nmi(truth, guess, n):
    if len(truth) == len(guess):
        I = 0.0
        for true_cluster in truth:
            for guess_cluster in guess:
                intersect_size = len(true_cluster.intersection(guess_cluster))
                if intersect_size == 0 or len(guess_cluster) == 0:
                    cur_I = 0.0
                else:
                    cur_I = (intersect_size / n) * \
                        np.log(n * intersect_size / (len(true_cluster) * len(guess_cluster)))
                I += cur_I

        H_truth = 0.0
        H_guess = 0.0
        for true_cluster in truth:
            H_truth -= (len(true_cluster) / n) * np.log(len(true_cluster) / n)
        for guess_cluster in guess:
            if len(guess_cluster) != 0:
                H_guess -= (len(guess_cluster) / n) * np.log(len(guess_cluster) / n)

        return (I / ((H_truth + H_guess) / 2))
    return 0.0

def _separate_pairs(clusters, n):
    positives, negatives = set(), set()
    for cluster in clusters:
        for node_A in cluster:
            for node_B in range(n):
                pair = (node_A, node_B)
                if node_B in cluster:
                    positives.add(pair)
                else:
                    negatives.add(pair)
    return positives, negatives

def calc_rand_ind(truth, guess, n, beta=None):
    truth_pos, truth_neg = _separate_pairs(truth, n)
    guess_pos, guess_neg = _separate_pairs(guess, n)

    tp = len(guess_pos.intersection(truth_pos))
    tn = len(guess_neg.intersection(truth_neg))

    fp = len(guess_pos.intersection(truth_neg))
    fn = len(guess_neg.intersection(truth_pos))

    if beta is None:
        return (tp + tn) / pairwise_guesses

    true_pos  = tp + fp
    guess_neg = tp + fn

    if true_pos == 0 and guess_neg == 0:
        return 0.0

    P = tp / true_pos
    R = tp / guess_neg
    return ((beta ** 2 + 1) * P * R) / (beta ** 2 * P + R)