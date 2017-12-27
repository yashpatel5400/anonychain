"""
__author__ = Yash Patel
__name__   = __init__.py
__description__ = Part that constructs the graph given the input data dump
"""

import subprocess
import struct

import networkx as nx
from networkx.drawing.nx_agraph import write_dot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def _read_in_chunks(file_object, chunk_size=9000):
    i=0
    while True:
        data = file_object.read(chunk_size)
        i += 1
        if i % 1000 == 0:
            print(i)
        
        if not data:
            break
        yield data

def plot_multi_graph(G):
    write_dot(G,'blockchain/multi_blockchain.dot')
    subprocess.call("./convert.sh", shell=True)

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
    return G

def create_visual_json(fn):
    f = open(fn, "rb")
    data  = {}
    data["nodes"] = []
    data["links"] = []
    nodes_to_ind = {}

    print("Parsing input binary dump...")
    for chunk in _read_in_chunks(f):
        # raw format: address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)
        for sequence_start in range(0, len(chunk), 9):
            sequence = chunk[sequence_start:sequence_start+9]
            address1ID, address2ID, heuristic = struct.unpack('iib', sequence)
            
            for addressID in [address1ID, address2ID]:
                if addressID not in nodes_to_ind:
                    nodes_to_ind[addressID] = len(data["nodes"])
                    data["nodes"].append({"id" : addressID})
                    
            data["links"].append({
                "source": nodes_to_ind[address1ID],
                "target": nodes_to_ind[address2ID],
                "weight": heuristic
            })

    with open("visualize/graph.json", "w") as dest:
        json.dump(data, dest)
    print("Produced visualization JSON!")
                    
if __name__ == "__main__":
    G = create_visual_json("blockchain/mini.dat")