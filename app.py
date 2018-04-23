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

from collections import defaultdict
import pickle
import time
import sys, getopt
import subprocess
from prettytable import PrettyTable

from algorithms import get_algorithms
from analysis.pca import plot_pca
from analysis.spectral import spectral_analysis, kmeans_analysis, cluster_analysis
from analysis.deanonymize import write_results, draw_results, calc_accuracy, calc_accuracies
from analysis.streaming import create_stream, streaming_analysis
from blockchain.read import get_data
from blockchain.metis import format_metis, run_metis
from coarsen.contract import contract_edges, contract_edges_matching, reconstruct_contracted
from setup.sbm import create_sbm, create_clusters

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
        "lib"             : "matplotlib",
        "multi_run"       : 1
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
            --gc <graph_coarsen> [(int) iterations of matchings found to be coarsened (default 0)]
            --lib                [('matplotlib','plotly') for plotting library]
            --mr                 [(int) indicates how many trials to be run in testing]"""

    opts, args = getopt.getopt(argv,"hr:b:d:w:c:n:g:p:q:m:",['lib=','cs=','gc=','mr='])
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
        elif opt in ("--gc"):  params["graph_coarsen"] = int(arg)
        elif opt in ("--lib"): params["lib"] = arg
        elif opt in ("--mr"):  params["multi_run"] = int(arg)

    if params["run_test"]:
        if params["cs"] is not None:
            params["cluster_sizes"] = [int(cluster) for cluster in params["cs"].split(",")]
        else:
            params["cluster_sizes"] = [cluster_size] * num_clusters
        params["clusters"] = create_clusters(params["cluster_sizes"])
    return params

def _pretty_format(d, header):
    t = PrettyTable(header)
    sorted_keys = sorted(d.keys())
    for key in sorted_keys:
       t.add_row([key, d[key]])
    return str(t)

def _update_accuracies(updates, purity, nmi, rand_ind, weighted_rand_ind, alg_name):
    purity[alg_name]            += updates["purity"]
    nmi[alg_name]               += updates["nmi"]
    rand_ind[alg_name]          += updates["rand_ind"]
    weighted_rand_ind[alg_name] += updates["weighted_ri"]

def main(argv):
    """Main application method that parses command line arguments and runs hierarchical
    and kmeans clustering. CMD-line arguments are specified in the help menu (run with -h).
    Final clustering outputs are provided in the output/ folder as eigen_guess and 
    kmeans_guess respectively. Intermediate results are available in output/eigen/

    Returns void
    """
    params = _cmd_graph(argv)
    produce_figures = True

    # algorithms to be used in the clustering runs (BOTH in testing and full analysis)
    to_run = set(["KMeans","MiniBatchKMeans","SpectralClustering"])
    
    if params["run_test"]:
        clusters = params["clusters"]
    else:
        clusters = None
        # change the line below if the remote source of the data is updated
        data_src = "https://s3.amazonaws.com/bitcoinclustering/cluster_data.dat"
        S, index_to_id = get_data(data_src, percent_bytes=params["byte_percent"])

    if params["run_test"]:
        purity            = defaultdict(lambda: 0.0)
        nmi               = defaultdict(lambda: 0.0)
        rand_ind          = defaultdict(lambda: 0.0)
        weighted_rand_ind = defaultdict(lambda: 0.0)
        accuracy_measures = [("purity",purity), ("nmi",nmi), 
            ("rand_ind",rand_ind), ("weighted_ri",weighted_rand_ind)]
        timeElapsed       = defaultdict(lambda: 0.0)

        for _ in range(params["multi_run"]):
            G = create_sbm(clusters, params["p"], params["q"], params["weighted"])
            if params["pca"]:
                plot_pca(G, clusters, plot_2d=True, plot_3d=True, plot_lib=params["lib"])

            n = sum([len(cluster) for cluster in clusters])
            weigh_edges = False
            if params["graph_coarsen"] is not None:
                params_fn = "p-{}_q-{}_gc-{}_n-{}".format(params["p"], 
                    params["q"], params["graph_coarsen"], n)
                num_clusters = len(clusters)

                # contracted_G, identified_nodes = contract_edges(G, num_edges=to_contract)
                contracted_G, identified_nodes = contract_edges_matching(G, 
                    num_iters=params["graph_coarsen"])
                print("Edges removed: {}".format(len(G.edges) - len(contracted_G.edges)))
                
                start = time.time()
                hier_cont_partitions = spectral_analysis(G, k=num_clusters)
                hier_partitions = reconstruct_contracted(identified_nodes, hier_cont_partitions)
                timeElapsed["ManualHierarchical"] += time.time() - start

                start = time.time()
                kmeans_cont_partitions = kmeans_analysis(G, k=num_clusters)
                kmeans_partitions = reconstruct_contracted(identified_nodes, kmeans_cont_partitions)
                timeElapsed["ManualKmeans"] += time.time() - start

                if produce_figures:
                    contracted_spring_pos = nx.spring_layout(contracted_G)
                    draw_results(contracted_G, contracted_spring_pos, hier_cont_partitions, 
                        "ManualHierarchical_cont_{}.png".format(params_fn), weigh_edges=weigh_edges)
                    draw_results(contracted_G, contracted_spring_pos, kmeans_cont_partitions, 
                        "ManualKmeans_cont_{}.png".format(params_fn), weigh_edges=weigh_edges)
                
            else:
                params_fn = "p-{}_q-{}_n-{}".format(params["p"], params["q"], n)
                if params["guess_clusters"]:
                    num_clusters = None
                else:
                    num_clusters = len(clusters)

                start = time.time()
                hier_partitions = spectral_analysis(G, k=num_clusters)
                timeElapsed["ManualHierarchical"] += time.time() - start

                start = time.time()
                kmeans_partitions = kmeans_analysis(G, k=num_clusters)
                timeElapsed["ManualKmeans"] += time.time() - start

            _update_accuracies(calc_accuracies(clusters, hier_partitions), 
                purity, nmi, rand_ind, weighted_rand_ind, "ManualHierarchical")
            _update_accuracies(calc_accuracies(clusters, kmeans_partitions), 
                purity, nmi, rand_ind, weighted_rand_ind, "ManualKmeans")
            
            if produce_figures:
                spring_pos  = nx.spring_layout(G)
                draw_results(G, spring_pos, clusters, 
                    "truth_{}.png".format(params_fn), weigh_edges=weigh_edges)
                draw_results(G, spring_pos, hier_partitions, 
                    "ManualHierarchical_{}.png".format(params_fn), weigh_edges=weigh_edges)
                draw_results(G, spring_pos, kmeans_partitions, 
                    "ManualKmeans_{}.png".format(params_fn), weigh_edges=weigh_edges)

            algorithms = get_algorithms(num_clusters)
            if params["graph_coarsen"] is not None:
                S = nx.adjacency_matrix(contracted_G)
            else:
                S = nx.adjacency_matrix(G)
            
            if params["run_metis"]:
                metis_fn = "output/test_metis.graph"
                format_metis(nx.adjacency_matrix(G), metis_fn)
                metis_partitions, time_elapsed = run_metis(metis_fn, num_clusters)

                _update_accuracies(calc_accuracies(clusters, metis_partitions), 
                    purity, nmi, rand_ind, weighted_rand_ind, "Metis")

                timeElapsed["Metis"] += time_elapsed
                if produce_figures:
                    draw_results(G, spring_pos, metis_partitions, 
                        "Metis_{}.png".format(params_fn), weigh_edges=weigh_edges)

            for alg_name in algorithms:
                if alg_name in to_run:
                    algorithm, args, kwds = algorithms[alg_name]
                    print(DELINEATION)
                    print("Running {} partitioning (coarsened: {})...".format(
                        alg_name, params["graph_coarsen"]))

                    start = time.time()
                    if params["graph_coarsen"] is not None:
                        cont_partitions = cluster_analysis(S, algorithm, args, kwds)
                        partitions = reconstruct_contracted(identified_nodes, cont_partitions)
                    else:
                        partitions = cluster_analysis(S, algorithm, args, kwds)
                    end = time.time()

                    if params["graph_coarsen"] is not None:
                        if produce_figures:
                            draw_results(contracted_G, contracted_spring_pos, cont_partitions, 
                                "{}_contracted_{}.png".format(alg_name, params_fn), 
                                weigh_edges=weigh_edges)

                    _update_accuracies(calc_accuracies(clusters, partitions), 
                        purity, nmi, rand_ind, weighted_rand_ind, alg_name)
                    timeElapsed[alg_name] += end - start
                    if produce_figures:
                        draw_results(G, spring_pos, partitions, 
                            "{}_{}.png".format(alg_name, params_fn), weigh_edges=weigh_edges)
                    print(DELINEATION)
        
        for accuracy_name, accuracies in accuracy_measures:
            for alg_name in accuracies.keys():
                accuracies[alg_name]  /= params["multi_run"]
                timeElapsed[alg_name] /= params["multi_run"]

            with open("output/{}_{}.txt".format(accuracy_name, params_fn),"w") as f:
                f.write(_pretty_format(accuracies,  ["algorithm","accuracy"]))
                f.write("\n")
                f.write(_pretty_format(timeElapsed, ["algorithm","time (s)"]))

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
    main(sys.argv[1:])