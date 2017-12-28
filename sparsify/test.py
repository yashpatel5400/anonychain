"""
__author__ = Yash Patel
__name__   = spectral.py
__description__ = Spectral sparsification of graphs
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from sparsify.spectral import SpectralSparsifier
from sparsify.sample import SampleSparsifier
from setup.sbm import create_sbm, create_clusters
from analysis.deanonymize import calc_accuracy, deanonymize

def _plot(G, fn):
    """Given an input graph and a filename, plots the graph at the file destination

    Returns void
    """
    print("Plotting {} graph...".format(fn))
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.axis('off')
    plt.savefig("output/sparsify/{}".format(fn))
    plt.close()

def _partition_graph(G, clusters):
    hier_partitions, kmeans_partitions = deanonymize(G, k=len(clusters))
    hier_accuracy = calc_accuracy(clusters, hier_partitions)
    kmeans_accuracy = calc_accuracy(clusters, kmeans_partitions)
    return hier_accuracy, kmeans_accuracy

def _run_test(G, clusters, sparsifier, sname):
    """Given an input graph, a sparsifier object (API of having a .sparsify(G) function),
    and a name of the sparsifier, plots the original and sparsifier versions of the graph
    in the output/sparsify/ folder

    Returns void
    """
    _plot(G, "{}_original".format(sname))
    orig_hier_accuracy, orig_kmeans_accuracy = _partition_graph(G, clusters)
    sparsifier.sparsify(G)
    new_hier_accuracy, new_kmeans_accuracy = _partition_graph(G, clusters)
    _plot(G, "{}_sparse".format(sname))
    
    print("Hierarchical -- Original: {} <=> New: {}".format(
        orig_hier_accuracy, new_hier_accuracy))
    print("K-means -- Original: {} <=> New: {}".format(
        orig_kmeans_accuracy, new_kmeans_accuracy))
    delta_hier = new_hier_accuracy - orig_hier_accuracy
    delta_kmeans = new_kmeans_accuracy - orig_kmeans_accuracy
    return delta_hier, delta_kmeans

def spectral_trial():
    epsilons = np.arange(0.25,5.0,.125)

    filtered_epsilons = []
    hier_deltas       = []
    kmeans_deltas     = []
    
    for epsilon in epsilons:
        try:
            cluster_sizes = [params["cluster_size"]] * params["num_clusters"]
            clusters = create_clusters(cluster_sizes)
            G = create_sbm(clusters, params["p"], params["q"], False)
            
            spectral_sparsifier = SpectralSparsifier(epsilon=epsilon)
            delta_hier, delta_kmeans = _run_test(G, clusters, spectral_sparsifier, "spectral")
            
            filtered_epsilons.append(epsilon)
            hier_deltas.append(delta_hier)
            kmeans_deltas.append(delta_kmeans)
            
        except:
            continue

    for acc_type, deltas in zip(["hierarchical","kmeans"],[hier_deltas,kmeans_deltas]):   
        plt.title("Epsilon vs. Drop in {} Accuracy".format(acc_type))
        plt.scatter(filtered_epsilons, deltas)
        plt.savefig("output/sparsify/{}_delta.png".format(acc_type))
        plt.close()

def sample_trial(params):
    cluster_sizes = [params["cluster_size"]] * params["num_clusters"]
    clusters = create_clusters(cluster_sizes)
    
    G = create_sbm(clusters, params["p"], params["q"], False)
    L = nx.normalized_laplacian_matrix(G).todense()
    w,_ = np.linalg.eig(L)
    sorted_w = sorted(w)
    
    sample_sparsifier = SampleSparsifier(sorted_w[len(clusters)], C=1)
    delta_hier, delta_kmeans = _run_test(G, clusters, sample_sparsifier, "sample")
    print("Sample-sparsifier accuracies: {} (hierarchical), " \
        "{} (kmeans)".format(delta_hier, delta_kmeans))

def main():
    params = {
        "p" : .75,
        "q" : .10,
        "cluster_size" : 5,
        "num_clusters" : 10
    }
    
    # spectral_trial(params)
    sample_trial(params)

if __name__ == "__main__":
    main()