"""
__author__ = Yash Patel
__name__   = app.py
__description__ = Main file for creating test SBM models and running clustering
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')

import sys, getopt
import subprocess

from setup.sbm import create_sbm, create_clusters
from analysis.pca import plot_pca
from analysis.deanonymize import draw_partitions, calc_accuracy, deanonymize

def _cmd_graph(argv):
    params = {
        "pca"             : "y",
        "p"               : 0.75,
        "q"               : 0.25,
        "guess_clusters"  : True ,
        "weighted"        : True,
        "cluster_size"    : 10,
        "num_clusters"    : 2,
        "cs"              : None,
        "lib"             : "matplotlib"
    }

    USAGE_STRING = """eigenvalues.py 
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
        opts, args = getopt.getopt(argv,"hd:c:n:g:p:q:w:",['lib=','cs='])
    except getopt.GetoptError:
        print("Using default values. To change use: \n{}".format(USAGE_STRING))

    for opt, arg in opts:
        if opt in ('-h'):
            print(USAGE_STRING)
            sys.exit()
        elif opt in ("-d"): params["pca"] = arg
        elif opt in ("-w"): params["weighted"] = (arg == "y")
        elif opt in ("-c"): params["cluster_size"] = int(arg)
        elif opt in ("-n"): params["num_clusters"] = int(arg)
        elif opt in ("-g"): params["guess_clusters"] = (arg == "y")
        
        elif opt in ("-p"): params["p"] = float(arg)
        elif opt in ("-q"): params["q"] = float(arg)
        
        elif opt in ("--cs"):  params["cs"] = arg
        elif opt in ("--lib"): params["lib"] = arg

    if params["cs"] is not None:
        params["cluster_sizes"] = [int(cluster) for cluster in params["cs"].split(",")]
    else:
        params["cluster_sizes"] = [cluster_size] * num_clusters

    params["clusters"] = create_clusters(params["cluster_sizes"])
    return params

def main(argv):
    params = _cmd_graph(argv)
    clusters = params["clusters"]
    sbm = create_sbm(clusters, params["p"], params["q"], params["weighted"])
    plot_pca(sbm, clusters, plot_2d=True, plot_3d=True, plot_lib=params["lib"])
    
    if params["guess_clusters"]:
        hier_partitions, kmeans_partitions = deanonymize(sbm, k=None)
    else: 
        hier_partitions, kmeans_partitions = deanonymize(sbm, k=len(clusters))
    
    print("hierarchical accuracy: {}".format(calc_accuracy(clusters, hier_partitions)))
    print("k-means accuracy: {}".format(calc_accuracy(clusters, kmeans_partitions)))

    spring_pos = nx.spring_layout(sbm)
    draw_partitions(sbm, spring_pos, clusters, clusters, "truth.png")
    draw_partitions(sbm, spring_pos, clusters, hier_partitions, "eigen_guess.png")
    draw_partitions(sbm, spring_pos, clusters, kmeans_partitions, "kmean_guess.png")

if __name__ == "__main__":
    print("Cleaning up directories...")
    subprocess.call("./clean.sh", shell=True)

    print("Performing SBM creation and segmentations...")
    main(sys.argv[1:])