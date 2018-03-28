"""
__author__ = Yash Patel
__name__   = app.py
__description__ = Main file for creating test SBM models and running clustering
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.cluster

import pickle
import time
import sys, getopt
import subprocess

from setup.sbm import create_sbm, create_clusters
from analysis.pca import plot_pca
from analysis.spectral import kmean_spectral, spectral_analysis_alt, cluster_analysis
from analysis.deanonymize import write_results, draw_results, calc_accuracy, deanonymize
from analysis.streaming import create_stream, streaming_analysis
from blockchain.read import get_data
from blockchain.metis import format_metis, run_metis
from coarsen.contract import contract_edges, reconstruct_contracted
from algorithms import get_algorithms

DELINEATION = "**********************************************************************"

def _cmd_graph(argv):
    """Parses arguments as specified by argv and returns as a dictionary. Entries
    are parsed as specified in the help menu (visible by running "python3 app.py -h")

    Returns parameters dictionary
    """
    params = {
        "byte_percent"    : .01,
        "cluster_size"    : 10,
        "pca"             : False,
        "guess_clusters"  : False,
        "run_metis"       : True,
        "num_clusters"    : 2,
        "run_test"        : True,
        "weighted"        : False,
        "p"               : 0.75,
        "q"               : 0.25,

        "cs"              : None,
        "graph_coarsen"   : None,
        "lib"             : "matplotlib"
    }

    USAGE_STRING = """eigenvalues.py 
            -b <byte_percent>    [(float) percent of bytes in full data to be analyzed]
            -c <cluster_size>    [(int) size of each cluster (assumed to be same for all)]
            -d <display_bool>    [(y/n) for whether to show PCA projections]
            -g <guess_bool>      [(y/n) to guess the number of clusters vs. take it as known] 
            -m <run_metis>       [(y/n) to additionally enable METIS clustering]
            -n <num_cluster>     [(int) number of clusters (distinct people)]
            -p <p_value>         [(0,1) float for in-cluster probability]
            -q <q_value>         [(0,1) float for non-cluster probability] 
            -r <run_test_bool>   [(y/n) for whether to create SBM to run test or run on actual data]
            -w <weighted_graph>  [(y/n) for whether to have weights on edges (randomized)]
            
            --cs <cluster_sizes> [(int list) size of each cluster (comma delimited)]
            --gc <graph_coarsen> [(0,1) percent of nodes to be coarsened (default 0)]
            --lib                [('matplotlib','plotly') for plotting library]"""

    opts, args = getopt.getopt(argv,"hr:b:d:w:c:n:g:p:q:m:",['lib=','cs=','gc='])
    for opt, arg in opts:
        if opt in ('-h'):
            print(USAGE_STRING)
            sys.exit()

        elif opt in ("-b"): params["byte_percent"]   = float(arg)
        elif opt in ("-c"): params["cluster_size"]   = int(arg)
        elif opt in ("-d"): params["pca"]            = (arg == "y")
        elif opt in ("-g"): params["guess_clusters"] = (arg == "y")
        elif opt in ("-m"): params["run_metis"]      = (arg == "y")
        elif opt in ("-n"): params["num_clusters"]   = int(arg)
        elif opt in ("-r"): params["run_test"]       = (arg == "y")
        elif opt in ("-w"): params["weighted"]       = (arg == "y")
        
        elif opt in ("-p"): params["p"] = float(arg)
        elif opt in ("-q"): params["q"] = float(arg)
        
        elif opt in ("--cs"):  params["cs"] = arg
        elif opt in ("--gc"):  params["graph_coarsen"] = float(arg)
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
    # algorithms to be used in the clustering runs (BOTH in testing and full analysis)
    to_run = set(["KMeans","MiniBatchKMeans","SpectralClustering"])
    
    if params["run_test"]:
        clusters = params["clusters"]
        G = create_sbm(clusters, params["p"], params["q"], params["weighted"])

        if params["pca"]:
            plot_pca(G, clusters, plot_2d=True, plot_3d=True, plot_lib=params["lib"])
    else:
        clusters = None
        # change the line below if the remote source of the data is updated
        data_src = "https://s3.amazonaws.com/bitcoinclustering/cluster_data.dat"
        S, index_to_id = get_data(data_src, percent_bytes=params["byte_percent"])

    if params["run_test"]:
        spring_pos  = nx.spring_layout(G)
        weigh_edges = False
        draw_results(G, spring_pos, clusters, "truth.png", weigh_edges=weigh_edges)

        if params["graph_coarsen"] is not None:
            num_clusters = len(clusters)
            to_contract = int(len(G.edges) * params["graph_coarsen"])
            contracted_G, identified_nodes = contract_edges(G, num_edges=to_contract)
            contracted_spring_pos = nx.spring_layout(contracted_G)

            hier_partitions, kmeans_partitions = deanonymize(contracted_G, k=num_clusters)

            draw_results(contracted_G, contracted_spring_pos, hier_partitions, 
                "ManualHierarchical_contracted.png", weigh_edges=weigh_edges)
            draw_results(contracted_G, contracted_spring_pos, kmeans_partitions, 
                "ManualKmeans_contracted.png", weigh_edges=weigh_edges)

            hier_partitions   = reconstruct_contracted(identified_nodes, hier_partitions)
            kmeans_partitions = reconstruct_contracted(identified_nodes, kmeans_partitions)
        else:
            if params["guess_clusters"]:
                hier_partitions, kmeans_partitions = deanonymize(G, k=None)
            else:
                num_clusters = len(clusters)
                hier_partitions, kmeans_partitions = deanonymize(G, k=num_clusters)

        print("Manual hierarchical accuracy: {}".format(
            calc_accuracy(clusters, hier_partitions)))
        print("Manual k-means accuracy: {}".format(
            calc_accuracy(clusters, kmeans_partitions)))
        
        draw_results(G, spring_pos, hier_partitions, 
            "ManualHierarchical_guess.png", weigh_edges=weigh_edges)
        draw_results(G, spring_pos, kmeans_partitions, 
            "ManualKmeans_guess.png", weigh_edges=weigh_edges)

        if not params["guess_clusters"]:
            algorithms = get_algorithms(num_clusters)

            if params["graph_coarsen"] is not None:
                S = nx.adjacency_matrix(contracted_G)
            else:
                S = nx.adjacency_matrix(G)
        
            if params["run_metis"]:
                metis_fn = "output/test_metis.graph"
                format_metis(nx.adjacency_matrix(G), metis_fn)
                metis_partitions = run_metis(metis_fn, num_clusters)

                print("Metis accuracy: {}".format(calc_accuracy(clusters, metis_partitions)))
                draw_results(G, spring_pos, metis_partitions, 
                    "metis_guess.png", weigh_edges=weigh_edges)

            for alg_name in algorithms:
                if alg_name in to_run:
                    algorithm, args, kwds = algorithms[alg_name]
                    print(DELINEATION)
                    print("Running {} partitioning (coarsened: {})...".format(
                        alg_name, params["graph_coarsen"]))

                    start = time.time()
                    partitions = cluster_analysis(S, algorithm, args, kwds)
                    if params["graph_coarsen"] is not None:
                        draw_results(contracted_G, contracted_spring_pos, partitions, 
                            "{}_contracted.png".format(alg_name), weigh_edges=weigh_edges)
                        partitions = reconstruct_contracted(identified_nodes, partitions)
                    end = time.time()

                    print("{} time elapsed (s): {}".format(alg_name, end - start))
                    print("{} accuracy: {}".format(alg_name, 
                        calc_accuracy(clusters, partitions)))
                    draw_results(G, spring_pos, partitions, 
                        "{}_guess.png".format(alg_name), weigh_edges=weigh_edges)
                    print(DELINEATION)
    else:
        num_clusters = params["num_clusters"]
        algorithms = get_algorithms(num_clusters)
        weigh_edges = False
        
        print("Creating NetworkX graph...")
        # G = nx.from_scipy_sparse_matrix(S)
        # spring_pos = nx.spring_layout(G)    

        for alg_name in algorithms:
            if alg_name in to_run:
                algorithm, args, kwds = algorithms[alg_name]
                print("Running {} partitioning...".format(alg_name))
                
                partitions = cluster_analysis(S, algorithm, args, kwds)
                write_results(partitions, index_to_id, "{}_guess".format(alg_name))
                # draw_results(G, spring_pos, partitions, 
                #     "{}_guess.png".format(alg_name), weigh_edges=weigh_edges)

        if params["run_metis"]:
            metis_fn = "blockchain/data_{0:f}.pickle".format(percent_bytes)
            metis_partitions = metis_from_pickle(metis_fn, num_clusters)

if __name__ == "__main__":
    print("Cleaning up directories...")
    subprocess.call("./clean.sh", shell=True)
    main(sys.argv[1:])