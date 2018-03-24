"""
__author__ = Yash Patel
__name__   = app.py
__description__ = Main file for creating test SBM models and running clustering
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys, getopt
import subprocess

from setup.sbm import create_sbm, create_clusters
from analysis.pca import plot_pca
from analysis.spectral import kmean_spectral, spectral_analysis_alt
from analysis.deanonymize import draw_partitions, calc_accuracy, deanonymize
from analysis.streaming import create_stream, streaming_analysis
from blockchain.read import create_simple_graph, create_similarity

def _cmd_graph(argv):
    """Parses arguments as specified by argv and returns as a dictionary. Entries
    are parsed as specified in the help menu (visible by running "python3 app.py -h")

    Returns parameters dictionary
    """
    params = {
        "run_test"        : True,
        "pca"             : True,
        "p"               : 0.75,
        "q"               : 0.25,
        "guess_clusters"  : False,
        "weighted"        : True,
        "cluster_size"    : 10,
        "num_clusters"    : 2,
        "cs"              : None,
        "lib"             : "matplotlib"
    }

    USAGE_STRING = """eigenvalues.py 
            -r <run_test_bool>   [(y/n) for whether to create SBM to run test or run on actual data]
            -d <display_bool>    [(y/n) for whether to show PCA projections]
            -w <weighted_graph>  [(y/n) for whether to have weights on edges (randomized)]
            -c <cluster_size>    [(int) size of each cluster (assumed to be same for all)]
            -n <num_cluster>     [(int) number of clusters (distinct people)]
            -g <guess_bool>      [(y/n) to guess the number of clusters vs. take it as known] 
            -p <p_value>         [(0,1) for in-cluster probability]
            -q <q_value>         [(0,1) for non-cluster probability] 
            --cs <cluster_sizes> [(int list) size of each cluster (comma delimited)]
            --lib                [('matplotlib','plotly') for plotting library]"""

    try:
        opts, args = getopt.getopt(argv,"hr:d:w:c:n:g:p:q:",['lib=','cs='])
    except getopt.GetoptError:
        print("Using default values. To change use: \n{}".format(USAGE_STRING))

    for opt, arg in opts:
        if opt in ('-h'):
            print(USAGE_STRING)
            sys.exit()
        elif opt in ("-r"): params["run_test"] = (arg == "y")
        elif opt in ("-d"): params["pca"] = (arg == "y")
        elif opt in ("-w"): params["weighted"] = (arg == "y")
        elif opt in ("-c"): params["cluster_size"] = int(arg)
        elif opt in ("-n"): params["num_clusters"] = int(arg)
        elif opt in ("-g"): params["guess_clusters"] = (arg == "y")
        
        elif opt in ("-p"): params["p"] = float(arg)
        elif opt in ("-q"): params["q"] = float(arg)
        
        elif opt in ("--cs"):  params["cs"] = arg
        elif opt in ("--lib"): params["lib"] = arg

    if params["run_test"]:
        if params["cs"] is not None:
            params["cluster_sizes"] = [int(cluster) for cluster in params["cs"].split(",")]
        else:
            params["cluster_sizes"] = [cluster_size] * num_clusters
        params["clusters"] = create_clusters(params["cluster_sizes"])
    return params

def main(argv):
    """Main application method that parses command line arguments and runs hierarchical
    and kmeans clustering. CMD-line arguments are specified in the help menu (run with -h).
    Final clustering outputs are provided in the output/ folder as eigen_guess and 
    kmeans_guess respectively. Intermediate results are available in output/eigen/

    Returns void
    """
    params = _cmd_graph(argv)

    if params["run_test"]:
        clusters = params["clusters"]
        G = create_sbm(clusters, params["p"], params["q"], params["weighted"])
        plot_pca(G, clusters, plot_2d=True, plot_3d=True, plot_lib=params["lib"])
    else:
        clusters = None
        L = create_similarity("blockchain/mini.dat")

    if params["run_test"]:
        num_clusters = len(clusters)
        if params["guess_clusters"]:
            hier_partitions, kmeans_partitions = deanonymize(G, k=None)
        else:
            hier_partitions, kmeans_partitions = deanonymize(G, k=num_clusters)

        spring_pos = nx.spring_layout(G)
        print("hierarchical accuracy: {}".format(calc_accuracy(clusters, hier_partitions)))
        print("k-means accuracy: {}".format(calc_accuracy(clusters, kmeans_partitions)))
        weigh_edges = False

        draw_partitions(G, spring_pos, clusters, "truth.png", weigh_edges=weigh_edges)
        draw_partitions(G, spring_pos, hier_partitions, 
            "eigen_guess.png", weigh_edges=weigh_edges)
        draw_partitions(G, spring_pos, kmeans_partitions, 
            "kmeans_guess.png", weigh_edges=weigh_edges)

    else:
        num_clusters = params["num_clusters"]
        weigh_edges = True
        kmeans_partitions_alt = spectral_analysis_alt(L, k=num_clusters)
        
        G = nx.from_scipy_sparse_matrix(L)
        spring_pos = nx.spring_layout(G)
        draw_partitions(G, spring_pos, kmeans_partitions_alt, 
            "kmeans_guess.png", weigh_edges=weigh_edges)

if __name__ == "__main__":
    print("Cleaning up directories...")
    subprocess.call("./clean.sh", shell=True)
    main(sys.argv[1:])