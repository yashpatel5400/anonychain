"""
__author__ = Yash Patel
__name__   = SpectralSparsifier.py
__description__ = Spectral sparsifier for graphs
"""

import numpy as np
import networkx as nx
import random

class SpectralSparsifier:
    def __init__(self, epsilon=.5):
        self.epsilon = epsilon

    def sparsify(self, G):
        n = nx.number_of_nodes(G)
        L = nx.laplacian_matrix(G).todense()
        L_pinv = np.linalg.pinv(L)

        expected_nonzero = int(6 * n * np.log(n) / (self.epsilon ** 2))
        print("Expected removals: {}".format(nx.number_of_edges(G) - expected_nonzero))

        rho = int(6 * np.log(n) / (self.epsilon ** 2))
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