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
    L_inv = np.linalg.inv(L)

    basis_vectors = np.eye(n)
    rho = int(6 * np.log(n) / (epsilon ** 2))

    to_remove = []
    for e in G.edges():
        u,v = e
        chi_e = basis_vectors[u] - basis_vectors[v]
        p_e = np.matmul(np.matmul(np.transpose(chi_e), L_inv), chi_e)
        w_e = 0
        for i in range(rho):
            Z_i = int(random.random() < p_e)
            w_e += Z_i / (rho * p_e)
        
        if w_e == 0:
            to_remove.append(e)

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
    G = nx.gnp_random_graph(25, 0.75)
    plot(G, "original")
    sparsify(G, epsilon=.05)
    plot(G, "sparse")

if __name__ == "__main__":
    main()