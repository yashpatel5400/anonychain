"""
__author__ = Yash Patel
__name__   = app.py
__description__ = Main file for running deanonymization on the BTC network
"""

import sys, getopt

from plot_pca import plot_pca
from sbm import create_sbm
from spectral import get_eigenvectors

def main(argv):
    pca          = "y"
    p            = 0.75
    q            = 0.25
    cluster_size = 10
    num_clusters = 2
    cs           = None
    lib          = "matplotlib"

    USAGE_STRING = """eigenvalues.py 
            -d <display_bool>    [(y/n) for whether to show PCA projections]
            -c <cluster_size>    [(int) size of each cluster (assumed to be same for all)]
            -n <num_cluster>     [(int) number of clusters (distinct people)]
            -p <p_value>         [(0,1) for in-cluster probability]
            -q <q_value>         [(0,1) for non-cluster probability] 
            --cs <cluster_sizes> [(int lisst) size of each cluster (comma delimited)]
            --lib                [('matplotlib','plotly') for plotting library]"""

    try:
        opts, args = getopt.getopt(argv,"d:c:n:p:q:",['lib=','cs='])
    except getopt.GetoptError:
        print("Using default values. To change use: \n{}".format(USAGE_STRING))

    for opt, arg in opts:
        if opt in ('-h'):
            print(USAGE_STRING)
            sys.exit()
        elif opt in ("-d"): pca = arg
        elif opt in ("-p"): p = float(arg)
        elif opt in ("-q"): q = float(arg)
        elif opt in ("-c"): cluster_size = int(arg)
        elif opt in ("-n"): num_clusters = int(arg)
        elif opt in ("--cs"):  cs = arg
        elif opt in ("--lib"): lib = arg

    if cs is not None:
        cluster_sizes = [int(cluster) for cluster in cs.split(",")]
    else:
        cluster_sizes = [cluster_size] * num_clusters
    
    sbm = create_sbm(cluster_sizes, p, q)
    if pca == "y":
        plot_pca(sbm, cluster_sizes, plot_2d=True, plot_3d=True, plot_lib=lib)
    get_eigenvectors(sbm)

if __name__ == "__main__":
    main(sys.argv[1:])