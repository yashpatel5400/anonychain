"""
__author__ = Yash Patel
__name__   = sample.py
__description__ = Sample-based sparsifier for graphs
"""

import numpy as np
import networkx as nx
import random

class SampleSparsifier:
    """Sample-based sparsifer class (https://arxiv.org/pdf/1711.01262.pdf)"""
    def __init__(self, lambda_k1, C=1.25, weighted=False):
        self.tau = C / lambda_k1
        self.weighted = weighted

    def sparsify(self, G):
        """Given an input graph, deletes edges from the graph (NOT the same
        as contracting edges). Action mutates original graph that is passed in

        Returns void
        """
        logn = np.log(nx.number_of_nodes(G))
        to_remove = []
        for e in G.edges():
            u,v = e
            if self.weighted:
                weight = G[u][v]['weight']
            else: weight = 1
            
            p_e = weight * self.tau * logn / G.degree(u)
            if random.random() > p_e:
                to_remove.append(e) 

        print("Sparsificiation complete: Deleted {} edges".format(len(to_remove)))
        for e in to_remove:
            G.remove_edge(*e)