"""
__author__ = Yash Patel
__name__   = streaming.py
__description__ = Runs spectral clustering for deanonymization on the BTC network,
in a streaming fashion (as opposed to batch processing as done in spectral.py). This
follows the implementation described in full by:
http://www.shivakasiviswanathan.com/ICDE16a.pdf
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_stream(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    rows, cols = adj_matrix.shape
    for i in range(rows):
        for j in range(cols):
            yield (i, j, adj_matrix[i, j])

def streaming_analysis(G, k):
    pass