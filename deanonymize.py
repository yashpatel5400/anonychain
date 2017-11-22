"""
__author__ = Yash Patel
__name__   = app.py
__description__ = Main file for running deanonymization on the BTC network
"""

import sys, getopt

from plot_pca import plot_pca
from sbm import create_sbm

def main(argv):
    pca = "y"
    p = 0.75
    q = 0.25
    lib = "plotly"

    USAGE_STRING = """eigenvalues.py 
            -d <display_bool> [(y/n) for whether to show PCA projections]
            -p <p_value>      [(0,1) for in-cluster probability]
            -q <q_value>      [(0,1) for non-cluster probability] 
            --lib             [('matplotlib','plotly') for plotting library]"""

    try:
        opts, args = getopt.getopt(argv,"d:p:q:",['lib='])
    except getopt.GetoptError:
        print("Using default values. To change use: \n{}".format(USAGE_STRING))

    for opt, arg in opts:
        if opt in ('-h'):
            print(USAGE_STRING)
            sys.exit()
        elif opt in ("-d"): pca = arg
        elif opt in ("-p"): p = float(arg)
        elif opt in ("-q"): q = float(arg)
        elif opt in ("--lib"): lib = arg

    cluster_sizes = [10,10]
    sbm = create_sbm(cluster_sizes, p, q)
    if pca == "y":
        plot_pca(sbm, cluster_sizes, plot_2d=True, plot_3d=True, plot_lib=lib)

if __name__ == "__main__":
    main(sys.argv[1:])