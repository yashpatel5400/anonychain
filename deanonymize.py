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
    cluster_sizes = [10,10,10]

    try:
        opts, args = getopt.getopt(argv,"d:p:q:")
    except getopt.GetoptError:
        print("Using default values. To change do: " \
            "eigenvalues.py -d <display_bool> -p <p_value> -q <q_value>")

    for opt, arg in opts:
        if opt == '-h':
            print('eigenvalues.py -d <display_bool> -p <p_value> -q <q_value>')
            sys.exit()
        elif opt in ("-d"):
            pca = arg
        elif opt in ("-p"):
            p = float(arg)
        elif opt in ("-q"):
            q = float(arg)

    MARGIN = .25
    sbm = create_sbm(cluster_sizes, p, q)
    if pca == "y":
        plot_pca(sbm, cluster_sizes, plot_2d=True, plot_3d=True, plot_lib="plotly")

if __name__ == "__main__":
    main(sys.argv[1:])