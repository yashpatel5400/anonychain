"""
__author__ = Yash Patel
__name__   = app.py
__description__ = Main file for creating test SBM models and running clustering
"""

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.cluster

import sys, getopt
import subprocess

from setup.sbm import create_sbm, create_clusters
from analysis.pca import plot_pca
from analysis.spectral import kmean_spectral, spectral_analysis_alt, cluster_analysis
from analysis.deanonymize import write_results, draw_results, calc_accuracy, deanonymize
from analysis.streaming import create_stream, streaming_analysis
from blockchain.read import get_data
from algorithms import get_algorithms

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
        "num_clusters"    : 2,
        "run_test"        : True,
        "weighted"        : True,
        "p"               : 0.75,
        "q"               : 0.25,
        "cs"              : None,
        "lib"             : "matplotlib"
    }

    USAGE_STRING = """eigenvalues.py 
            -b <byte_percent>    [(float) percent of bytes in full data to be analyzed]
            -c <cluster_size>    [(int) size of each cluster (assumed to be same for all)]
            -d <display_bool>    [(y/n) for whether to show PCA projections]
            -g <guess_bool>      [(y/n) to guess the number of clusters vs. take it as known] 
            -n <num_cluster>     [(int) number of clusters (distinct people)]
            -p <p_value>         [(0,1) float for in-cluster probability]
            -q <q_value>         [(0,1) float for non-cluster probability] 
            -r <run_test_bool>   [(y/n) for whether to create SBM to run test or run on actual data]
            -w <weighted_graph>  [(y/n) for whether to have weights on edges (randomized)]
            --cs <cluster_sizes> [(int list) size of each cluster (comma delimited)]
            --lib                [('matplotlib','plotly') for plotting library]"""

    opts, args = getopt.getopt(argv,"hr:b:d:w:c:n:g:p:q:",['lib=','cs='])
    for opt, arg in opts:
        if opt in ('-h'):
            print(USAGE_STRING)
            sys.exit()

        elif opt in ("-b"): params["byte_percent"]   = float(arg)
        elif opt in ("-c"): params["cluster_size"]   = int(arg)
        elif opt in ("-d"): params["pca"]            = (arg == "y")
        elif opt in ("-g"): params["guess_clusters"] = (arg == "y")
        elif opt in ("-n"): params["num_clusters"]   = int(arg)
        elif opt in ("-r"): params["run_test"]       = (arg == "y")
        elif opt in ("-w"): params["weighted"]       = (arg == "y")
        
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
        L, index_to_id = get_data(data_src, percent_bytes=params["byte_percent"])

    if params["run_test"]:
        num_clusters = len(clusters)
        algorithms = get_algorithms(num_clusters)

        spring_pos  = nx.spring_layout(G)
        weigh_edges = False
        draw_results(G, spring_pos, clusters, "truth.png", weigh_edges=weigh_edges)

        if params["guess_clusters"]:
            hier_partitions, kmeans_partitions = deanonymize(G, k=None)
            print("hierarchical accuracy: {}".format(calc_accuracy(clusters, hier_partitions)))
            print("k-means accuracy: {}".format(calc_accuracy(clusters, kmeans_partitions)))
            
            draw_results(G, spring_pos, hier_partitions, 
                "eigen_guess.png", weigh_edges=weigh_edges)
            draw_results(G, spring_pos, kmeans_partitions, 
                "kmeans_guess.png", weigh_edges=weigh_edges)
        else:
            for alg_name in algorithms:
                if alg_name in to_run:
                    algorithm, args, kwds = algorithms[alg_name]
                    print("Running {} partitioning...".format(alg_name))
                    
                    L = nx.laplacian_matrix(G)
                    partitions = cluster_analysis(L, algorithm, args, kwds)
                    
                    print("{} accuracy: {}".format(alg_name, 
                        calc_accuracy(clusters, partitions)))
                    draw_results(G, spring_pos, partitions, 
                        "{}_guess.png".format(alg_name), weigh_edges=weigh_edges)
    else:
        num_clusters = params["num_clusters"]
        algorithms = get_algorithms(num_clusters)
        weigh_edges = False
        
        print("Creating NetworkX graph...")
        # G = nx.from_scipy_sparse_matrix(L)
        # spring_pos = nx.spring_layout(G)    

        for alg_name in algorithms:
            if alg_name in to_run:
                algorithm, args, kwds = algorithms[alg_name]
                print("Running {} partitioning...".format(alg_name))
                
                partitions = cluster_analysis(L, algorithm, args, kwds)
                write_results(partitions, index_to_id, "{}_guess".format(alg_name))
                # draw_results(G, spring_pos, partitions, 
                #     "{}_guess.png".format(alg_name), weigh_edges=weigh_edges)

if __name__ == "__main__":
    print("Cleaning up directories...")
    subprocess.call("./clean.sh", shell=True)
    main(sys.argv[1:])