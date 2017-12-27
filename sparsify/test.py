"""
__author__ = Yash Patel
__name__   = spectral.py
__description__ = Spectral sparsification of graphs
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

from sparsify.spectral import SpectralSparsifier

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

def _run_test(G, sparsifier, sname):
    """Given an input graph, a sparsifier object (API of having a .sparsify(G) function),
    and a name of the sparsifier, plots the original and sparsifier versions of the graph
    in the output/sparsify/ folder

    Returns void
    """
    _plot(G, "{}_original".format(sname))
    sparsifier.sparsify(G)
    _plot(G, "{}_sparse".format(sname))

def main():
    G = nx.binomial_graph(250, .25)
    spectral_sparsifier = SpectralSparsifier(epsilon=2.5)
    _run_test(G, spectral_sparsifier, "spectral")

if __name__ == "__main__":
    main()