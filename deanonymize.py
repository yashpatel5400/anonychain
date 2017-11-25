"""
__author__ = Yash Patel
__name__   = app.py
__description__ = Main file for running deanonymization on the BTC network
"""

import sys, getopt
import networkx as nx

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from plot_pca import plot_pca
from sbm import create_sbm
from spectral import spectral_analysis, kmeans_analysis
from constants import colors

def _create_clusters(cluster_sizes):
    completed_nodes = 0
    clusters = []
    for cluster_size in cluster_sizes:
        clusters.append(set(range(completed_nodes, completed_nodes + cluster_size)))
        completed_nodes += cluster_size
    return clusters

def _draw_partitions(G, pos, clusters, partitions, fn, calc_accuracy=True):
    if calc_accuracy and len(clusters) == len(partitions):
        partitions = _reorder_clusters(clusters, partitions)
        print("{} accuracy: {}%".format(fn, _calc_accuracy(clusters, partitions)))

    guessed_colors = [colors[j] for i in range(len(G.nodes))
        for j, partition in enumerate(partitions) if i in partition]

    nx.draw(G, pos, node_size=100, node_color=guessed_colors)
    plt.savefig("output/{}".format(fn))
    plt.close()

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

def _calc_accuracy(truth, guess):
    num_correct = 0
    total_nodes = 0
    for i in range(len(truth)):
        num_correct += len(truth[i].intersection(guess[i]))
        total_nodes += len(truth[i])
    return 100.0 * (num_correct/total_nodes)

def _cmd_graph(argv):
    pca             = "y"
    p               = 0.75
    q               = 0.25

    assume_clusters = False
    cluster_size    = 10
    num_clusters    = 2
    
    cs              = None
    lib             = "matplotlib"

    USAGE_STRING = """eigenvalues.py 
            -d <display_bool>    [(y/n) for whether to show PCA projections]
            -c <cluster_size>    [(int) size of each cluster (assumed to be same for all)]
            -n <num_cluster>     [(int) number of clusters (distinct people)]
            -g <guess_bool>      [(y/n) to guess the number of clusters vs. take it as known] 
            -p <p_value>         [(0,1) for in-cluster probability]
            -q <q_value>         [(0,1) for non-cluster probability] 
            --cs <cluster_sizes> [(int lisst) size of each cluster (comma delimited)]
            --lib                [('matplotlib','plotly') for plotting library]"""

    try:
        opts, args = getopt.getopt(argv,"d:c:n:g:p:q:",['lib=','cs='])
    except getopt.GetoptError:
        print("Using default values. To change use: \n{}".format(USAGE_STRING))

    for opt, arg in opts:
        if opt in ('-h'):
            print(USAGE_STRING)
            sys.exit()
        elif opt in ("-d"): pca = arg
        elif opt in ("-c"): cluster_size = int(arg)
        elif opt in ("-n"): num_clusters = int(arg)
        elif opt in ("-g"): assume_clusters = (arg == "y")
        
        elif opt in ("-p"): p = float(arg)
        elif opt in ("-q"): q = float(arg)
        
        elif opt in ("--cs"):  cs = arg
        elif opt in ("--lib"): lib = arg

    if cs is not None:
        cluster_sizes = [int(cluster) for cluster in cs.split(",")]
    else:
        cluster_sizes = [cluster_size] * num_clusters

    clusters = _create_clusters(cluster_sizes)
    return clusters, assume_clusters, pca, p, q, lib

def main(argv):
    clusters, assume_clusters, pca, p, q, lib = _cmd_graph(argv)
    sbm = create_sbm(clusters, p, q)

    if pca == "y":
        plot_pca(sbm, clusters, plot_2d=True, plot_3d=True, plot_lib=lib)
    
    if assume_clusters:
        partitions = spectral_analysis(sbm, k=len(clusters), normalize=True)
    else: partitions = spectral_analysis(sbm, k=None, normalize=True)

    kmeans_analysis(sbm, clusters, len(clusters))

    spring_pos = nx.spring_layout(sbm)
    _draw_partitions(sbm, spring_pos, clusters, clusters,   "truth.png", calc_accuracy=False)
    _draw_partitions(sbm, spring_pos, clusters, partitions, "guess.png", calc_accuracy=True)

if __name__ == "__main__":
    main(sys.argv[1:])