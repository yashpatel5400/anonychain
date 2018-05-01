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
import time

from sparsify.spectral import SpectralSparsifier
from sparsify.sample import SampleSparsifier
from setup.sbm import create_sbm, create_clusters
from analysis.deanonymize import calc_accuracies, spectral_analysis, kmeans_analysis

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
    start = time.time()
    hier_partitions = spectral_analysis(G, k=len(clusters))
    hier_accuracies = calc_accuracies(clusters, hier_partitions)
    end = time.time()
    print(end-start)
    hier_accuracies["time"] = end - start

    start = time.time()
    kmeans_partitions = kmeans_analysis(G, k=len(clusters))
    kmeans_accuracies = calc_accuracies(clusters, kmeans_partitions)
    kmeans_accuracies["time"] = time.time() - start
    return hier_accuracies, kmeans_accuracies

def _run_test(G, clusters, sparsifier, param, sname):
    """Given an input graph, a sparsifier object (API of having a .sparsify(G) function),
    and a name of the sparsifier, plots the original and sparsifier versions of the graph
    in the output/sparsify/ folder

    Returns void
    """
    _plot(G, "{}_{}_original.png".format(sname, param))
    orig_hier_accuracies, orig_kmeans_accuracies = _partition_graph(G, clusters)
    sparsifier.sparsify(G)
    new_hier_accuracies, new_kmeans_accuracies = _partition_graph(G, clusters)
    _plot(G, "{}_{}_sparse.png".format(sname, param))
    
    delta_hier = {}
    delta_kmeans = {}

    for accuracy_metric in orig_hier_accuracies:
        if accuracy_metric != "time":
            delta_hier[accuracy_metric] = new_hier_accuracies[accuracy_metric] - \
                orig_hier_accuracies[accuracy_metric]
            delta_kmeans[accuracy_metric] = new_kmeans_accuracies[accuracy_metric] - \
                orig_kmeans_accuracies[accuracy_metric]
        else:
            delta_hier[accuracy_metric] = new_hier_accuracies[accuracy_metric]
            delta_kmeans[accuracy_metric] = new_kmeans_accuracies[accuracy_metric]

    return delta_hier, delta_kmeans

def _plot_trial_results(params, hier_deltas, kmeans_deltas, trial_type):
    params = np.array(params)
    for acc_type, deltas in zip(["hierarchical","kmeans"],[hier_deltas,kmeans_deltas]):
        metrics = {}
        for trial in deltas:
            for accuracy_metric in trial:
                if accuracy_metric not in metrics:
                    metrics[accuracy_metric] = []
                metrics[accuracy_metric].append(trial[accuracy_metric])

        print(metrics)
        for accuracy_metric in metrics:
            plt.title("Param vs. Drop in {} {}".format(acc_type, accuracy_metric))
            
            m, b = np.polyfit(params, metrics[accuracy_metric], 1)
            plt.scatter(params, metrics[accuracy_metric])
            plt.plot(params, m * params + b, '-')

            plt.savefig("output/sparsify/{}_{}_{}.png".format(accuracy_metric, acc_type,trial_type))
            plt.close()

def spectral_trial(params):
    epsilons = np.arange(0.25,10.0,.125)

    filtered_epsilons = []
    hier_deltas       = []
    kmeans_deltas     = []
    
    for epsilon in epsilons:
        try:
            cluster_sizes = [params["cluster_size"]] * params["num_clusters"]
            clusters = create_clusters(cluster_sizes)
            G = create_sbm(clusters, params["p"], params["q"], False)
                
            spectral_sparsifier = SpectralSparsifier(epsilon=epsilon)
            delta_hier, delta_kmeans = _run_test(G, clusters, 
                spectral_sparsifier, epsilon, "spectral")
                
            filtered_epsilons.append(epsilon)
            hier_deltas.append(delta_hier)
            kmeans_deltas.append(delta_kmeans)
            
        except:
            continue
    _plot_trial_results(filtered_epsilons,hier_deltas,kmeans_deltas,"spectral")

def sample_trial(params):
    Cs = np.arange(0.25,.50,.25)

    filtered_Cs   = []
    hier_deltas   = []
    kmeans_deltas = []
    
    for C in Cs:
        try:
            cluster_sizes = [params["cluster_size"]] * params["num_clusters"]
            clusters = create_clusters(cluster_sizes)
            
            G = create_sbm(clusters, params["p"], params["q"], False)
            L = nx.normalized_laplacian_matrix(G).todense()
            w,_ = np.linalg.eig(L)
            sorted_w = sorted(w)
            
            sample_sparsifier = SampleSparsifier(sorted_w[len(clusters)], C=C)
            delta_hier, delta_kmeans = _run_test(G, clusters, 
                sample_sparsifier, C, "sample")
            
            filtered_Cs.append(C)
            hier_deltas.append(delta_hier)
            kmeans_deltas.append(delta_kmeans)

        except:
            continue
    _plot_trial_results(filtered_Cs,hier_deltas,kmeans_deltas,"sample")

def main():
    params = {
        "p" : .75,
        "q" : .10,
        "cluster_size" : 10,
        "num_clusters" : 10
    }
    
    spectral_trial(params)
    
if __name__ == "__main__":
    main()