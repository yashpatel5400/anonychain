"""
__author__ = Yash Patel
__name__   = streaming.py
__description__ = Runs spectral clustering for deanonymization on the BTC network,
in a streaming fashion (as opposed to batch processing as done in spectral.py). This
follows the implementation described in full by:
http://www.shivakasiviswanathan.com/ICDE16a.pdf
"""

import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_stream(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    rows, cols = adj_matrix.shape
    for i in range(rows):
        for j in range(cols):
            yield (i, j, adj_matrix[i, j])

def streamSC(n, k):
    """
    streaming spectral clustering implementation used for clustering each of the
    n datapoints into k clusters
    """
    m = 1 # dimension of feature space : nothing other than a numerical weight for us
    l = None

    f = 1 / (k * (1 + math.log(n)))
    s = np.zeros(m)
    B = np.zeros((m, l))

def _streamEMB():
    """
    helper function for streaming SC used for embedding
    """
    pass

def _streamKMstep():
    """
    helper function for streaming SC used for KM clustering
    """
    pass


def streaming_analysis(stream):
    for edge in stream:
        v1, v2, weight = edge