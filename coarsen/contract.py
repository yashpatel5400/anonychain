"""
__author__ = Yash Patel
__name__   = contract.py
__description__ = Contraction script used for testing the effectiveness of 
uniformly at random edge contractions in retaining clustering accuracy
"""

import numpy as np
import networkx as nx
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from setup.sbm import create_sbm, create_clusters
from analysis.pca import plot_pca
from analysis.deanonymize import draw_results, calc_accuracy, deanonymize

def contract_edges(G, num_edges):
    """Given a graph G and a desired number of edges to be contracted, contracts edges
    uniformly at random (non-mutating of the original graph). Edges are contracted such
    that the two endpoints are now "identified" with one another. This mapping is returned
    as a dictionary If more edges are provided than can be contracted, an error is thrown. 

    Returns (1) contracted graph (NetworkX Graph); 
    (2) identified nodes dictionary (NetworkX node -> NetworkX node)
    """
    identified_nodes = {}
    for _ in range(num_edges):
        edges = list(G.edges)
        random_edge = edges[random.randint(0,len(edges)-1)]
        node_to_contract = random_edge[1]
        identified_nodes[node_to_contract] = random_edge[0] # right gets contracted into left
        G = nx.contracted_edge(G, random_edge, self_loops=False)
    return G, identified_nodes

def reconstruct_contracted(identified_nodes, partitions):
    """Given the node identifications from the original graph contraction and the 
    partitions formed on the contracted graph, creates partitions on the original graph
    by associating the contracted nodes with the partitions of their "partners"

    Returns partitions of the original graph (list of lists of ints)
    """
    for contracted in identified_nodes:
        for partition in partitions:
            if identified_nodes[contracted] in partition:
                partition.add(contracted)
                break
    return partitions

def plot_graph(G, G_type):
    """Given the original graph G and contracted graph, plots both in the output/contraction/
    folder, named respectively. 

    Returns void
    """
    pos = nx.spring_layout(G)
    nx.draw(G, contracted_pos)
    plt.axis('off')
    plt.savefig("output/contraction/{}.png".format(G_type))
    plt.close()

def contract_deanonymize(G, k, to_contract, to_plot=False):
    """Given graph G, the number of clusters k, the number of edges to be contracted, and
    whether the output is to be plotted or not, runs clustering on the graph and produces
    output in the output/contraction/ folder

    Returns (1) hierarchical partitions (list of lists of ints); 
    (2) kmeans partitions (list of lists of ints)
    """
    contracted_G, identified_nodes = contract_edges(G, num_edges=to_contract)

    hier_partitions, kmeans_partitions = deanonymize(contracted_G, k=k)
    hier_partitions   = reconstruct_contracted(identified_nodes, hier_partitions)
    kmeans_partitions = reconstruct_contracted(identified_nodes, kmeans_partitions)

    if to_plot:
        print("Plotting graphs...")
        plot_graph(G, "original")
        plot_graph(contracted_G, "contracted")

        spring_pos = nx.spring_layout(G)
        draw_partitions(G, spring_pos, clusters, 
            "contraction/truth.png", weigh_edges=False)
        draw_partitions(G, spring_pos, hier_partitions, 
            "contraction/eigen_guess.png", weigh_edges=False)
        draw_partitions(G, spring_pos, kmeans_partitions, 
            "contraction/kmeans_guess.png", weigh_edges=False)
    return hier_partitions, kmeans_partitions

def single_contract_test(params):
    """Given graph parameters with p (probability in-cluster), 
    q (probability out-of-cluster), and percent_edges (percent of total edges
    to be contracted), runs a with-contraction clustering trial on a randomly
    generated SBM (Stochastic Block Model) graph

    Returns (1) hierarchical accuracy (float); (2) kmeans accuracy (float)
    """
    cluster_size = 8
    num_clusters = 5
    cluster_sizes = [cluster_size] * num_clusters
    clusters = create_clusters(cluster_sizes)

    G = create_sbm(clusters, params["p"], params["q"], False)
    to_contract = int(len(G.edges) * params["percent_edges"])

    num_clusters = len(clusters)
    hier_partitions, kmeans_partitions = contract_deanonymize(G, 
        k=num_clusters, to_contract=to_contract)
    
    hier_accuracy   = calc_accuracy(clusters, hier_partitions)
    kmeans_accuracy = calc_accuracy(clusters, kmeans_partitions)

    print("hierarchical accuracy: {}".format(hier_accuracy))
    print("k-means accuracy: {}".format(kmeans_accuracy))
    return hier_accuracy, kmeans_accuracy

def contract_tests():
    """Runs tests for all values of p between [0,1) (.1 increments), q between
    [0,p) (.1 increments), and percent_edges [0,.3) (.03 increments)

    Returns void
    """
    edge_percents = np.arange(0, .30, 0.03)
    num_trials    = 10
    params        = {}

    for p in np.arange(0, 1.0, 0.1):
        for q in np.arange(0, p, 0.1):
            hier_accuracies   = []
            kmeans_accuracies = []

            for percent_edges in edge_percents:
                hier_trial   = []
                kmeans_trial = []
                
                params["p"] = p
                params["q"] = q
                params["percent_edges"] = percent_edges

                for trial in range(num_trials):
                    try: 
                        hier_accuracy, kmeans_accuracy = single_contract_test(params)
                        hier_trial.append(hier_accuracy)
                        kmeans_trial.append(kmeans_accuracy)
                    except:
                        continue

                hier_accuracies.append(np.median(hier_trial))
                kmeans_accuracies.append(np.median(kmeans_trial))

            for graph_type, accuracy in \
                zip(["hierarchical","kmeans"], [hier_accuracies,kmeans_accuracies]):

                plt.title("{} {} {}".format(graph_type,p,q))
                plt.plot(edge_percents, accuracy)
                plt.savefig("output/contraction/{}_{}_{}.png".format(
                    graph_type,round(p,2),round(q,2)))
                plt.close()

if __name__ == "__main__":
    contract_tests()