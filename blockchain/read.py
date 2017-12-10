"""
__author__ = Yash Patel
__name__   = __init__.py
__description__ = Part that constructs the graph given the input data dump
"""

import subprocess

import networkx as nx
from networkx.drawing.nx_agraph import write_dot

def _read_in_chunks(file_object, chunk_size=900):
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def _plot_graph(G):
    write_dot(G,'multi.dot')
    subprocess.call("./convert.sh", shell=True)

def create_graph(fn):
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
    _plot_graph(G)
    return G

if __name__ == "__main__":
    G = create_graph("graph.dat")