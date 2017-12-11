"""
__author__ = Yash Patel
__name__   = __init__.py
__description__ = Part that constructs the graph given the input data dump
"""

import subprocess

import networkx as nx
from networkx.drawing.nx_agraph import write_dot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def _read_in_chunks(file_object, chunk_size=900):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def _plot_multi_graph(G):
    write_dot(G,'blockchain/multi_blockchain.dot')
    subprocess.call("./convert.sh", shell=True)

def _plot_simple_graph(G):
    edgewidth = [d['weight'] for (u,v,d) in G.edges(data=True)]
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, width=edgewidth)

    plt.axis('off')
    plt.savefig("blockchain/simple_blockchain.png")
    plt.close()

def create_multi_graph(fn):
    print("Reading blockchain graph as multi graph...")
    G = nx.MultiGraph()
    f = open(fn, "rb")
    
    nodes = set()
    for chunk in _read_in_chunks(f):
        # raw format: address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)
        for sequence_start in range(0, len(chunk), 9):
            address1ID = chunk[sequence_start:(sequence_start+4)] 
            address2ID = chunk[(sequence_start+4):(sequence_start+8)]
            heuristic  = chunk[sequence_start+8]
            if address1ID not in nodes:
                G.add_node(address1ID)
                nodes.add(address1ID)
            if address1ID not in nodes:
                G.add_node(address2ID)
                nodes.add(address2ID)
            G.add_edge(address1ID, address2ID, heuristic=heuristic)
    _plot_multi_graph(G)
    return G

def create_simple_graph(fn):
    print("Reading blockchain graph as simple graph...")
    G = nx.MultiGraph()
    f = open(fn, "rb")

    for chunk in _read_in_chunks(f):
        # raw format: address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)
        for sequence_start in range(0, len(chunk), 9):
            address1ID = chunk[sequence_start:(sequence_start+4)] 
            address2ID = chunk[(sequence_start+4):(sequence_start+8)]
            heuristic  = chunk[sequence_start+8]
            if G.has_edge(address1ID, address2ID):
                G.add_edge(address1ID, address2ID, 
                    weight=G[address1ID][address2ID][0]["weight"] + 1)
            else:
                G.add_edge(address1ID, address2ID, weight=1)
    _plot_simple_graph(G)
    return G

if __name__ == "__main__":
    G = create_simple_graph("graph.dat")