"""
__author__ = Yash Patel
__name__   = spectral.py
__description__ = Spectral sparsification of graphs
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

def sparsify(G, epsilon=.5):
    # algorithm documented in https://www.cs.ubc.ca/~nickhar/Cargese3.pdf
    n = nx.number_of_nodes(G)
    L = nx.laplacian_matrix(G).todense()
    L_pinv = np.linalg.pinv(L)

    expected_nonzero = int(6 * n * np.log(n) / (epsilon ** 2))
    print("Expected removals: {}".format(nx.number_of_edges(G) - expected_nonzero))

    rho = int(6 * np.log(n) / (epsilon ** 2))
    to_remove = []
    for e in G.edges():
        u,v = e
        p_e = L_pinv[u,u] + L_pinv[v,v] - 2 * L_pinv[u,v]
        w_e = 0

        for i in range(rho):
            Z_i = int(random.random() < p_e)
            w_e += Z_i / (rho * p_e)
        if w_e == 0:
            to_remove.append(e)

    print("Sparsificiation complete: Deleted {} edges".format(len(to_remove)))
    for e in to_remove:
        G.remove_edge(*e)

def plot(G, fn):
    print("Plotting {} graph...".format(fn))
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    plt.axis('off')
    plt.savefig("output/sparsify/{}".format(fn))
    plt.close()

def main():
    G = nx.binomial_graph(250, .25)
    plot(G, "original")
    sparsify(G, epsilon=.5)
    plot(G, "sparse")

if __name__ == "__main__":
    main()