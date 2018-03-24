"""
__author__ = Yash Patel
__name__   = __init__.py
__description__ = Part that constructs the graph given the input data dump
"""

import subprocess
import struct

import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import write_dot
from scipy.sparse import dok_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def _read_in_chunks(file_object, chunk_size=9000):
    """Given a file pointer and a chunk size, yields an iterator over the file
    contents to avoid having to read it all into memory

    Returns void
    """
    i=0
    while True:
        data = file_object.read(chunk_size)
        i += 1
        if i % 1000 == 0:
            print(i)
        
        if not data:
            break
        yield data

def _plot_multi_graph(G):
    """Given a multigraph G, produces a corresponding visualization

    Returns void
    """
    write_dot(G,'blockchain/multi_blockchain.dot')
    subprocess.call("./convert.sh", shell=True)      

def _create_multi_graph(fn):
    """Given an input filename, constructs an unweighted multigraph with edges labelled
    with an additional "heuristic" property. The input data MUST be specified as 
    follows (no separators):

    address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)

    Returns Multigraph (NetworkX object)
    """
    print("Reading blockchain graph as multi graph...")
    G = nx.MultiGraph()
    f = open(fn, "rb")
    
    nodes = set()
    for chunk in _read_in_chunks(f):
        # raw format: address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)
        for sequence_start in range(0, len(chunk), 9):
            sequence = chunk[sequence_start:sequence_start+9]
            address1ID, address2ID, heuristic = struct.unpack('iib', sequence)
            for addressID in [address1ID, address2ID]:
                if addressID not in nodes:
                    G.add_node(addressID)
                    nodes.add(addressID)
            G.add_edge(address1ID, address2ID, heuristic=heuristic)
    return G

def _create_simple_graph(fn):
    """Given an input filename, constructs a weighted, undirected graph. 
    The input data MUST be specified as follows (no separators):

    address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)

    Returns Simple, weighted graph (NetworkX object)
    """
    print("Reading blockchain graph as simple graph...")
    G = nx.MultiGraph()
    f = open(fn, "rb")

    for chunk in _read_in_chunks(f):
        for sequence_start in range(0, len(chunk), 9):
            # raw format: address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)
            sequence = chunk[sequence_start:sequence_start+9]
            address1ID, address2ID, heuristic = struct.unpack('iib', sequence)
            if G.has_edge(address1ID, address2ID):
                G.add_edge(address1ID, address2ID, 
                    weight=G[address1ID][address2ID][0]["weight"] + 1)
            else:
                G.add_edge(address1ID, address2ID, weight=1)
    return G

def _count_nodes(fn):
    f = open(fn, "rb")
    id_to_index = {}
    for chunk in _read_in_chunks(f):
        for sequence_start in range(0, len(chunk), 9):
            sequence = chunk[sequence_start:sequence_start+9]
            address1ID, address2ID, heuristic = struct.unpack('iib', sequence)

            if address1ID not in id_to_index:
                id_to_index[address1ID] = len(id_to_index)
            if address2ID not in id_to_index:
                id_to_index[address2ID] = len(id_to_index)
    return len(id_to_index)  

def _create_similarity(fn, size):
    """Given an input filename, constructs the similarity matrix for the associated
    graph. NetworkX is NOT used directly for purposes of space efficiency.
    The input data MUST be specified as follows (no separators):

    address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)

    Returns scipy-sparse matrix
    """
    print("Reading blockchain graph as sparse similarity matrix...")
    
    S = dok_matrix((size, size), dtype=np.float32)
    f = open(fn, "rb")

    id_to_index = {}
    for chunk in _read_in_chunks(f):
        for sequence_start in range(0, len(chunk), 9):
            sequence = chunk[sequence_start:sequence_start+9]
            address1ID, address2ID, heuristic = struct.unpack('iib', sequence)

            if address1ID not in id_to_index:
                id_to_index[address1ID] = len(id_to_index)
            if address2ID not in id_to_index:
                id_to_index[address2ID] = len(id_to_index)

            address1Index = id_to_index[address1ID]
            address2Index = id_to_index[address2ID]

            S[address1Index, address2Index] += 1
            S[address2Index, address1Index] += 1
    return S
    
def _create_visual_json(fn):
    """Given an input filename, reads the file and outputs the corresponding JSON formatted
    data to be visualized on the HTML visualization page. The input data MUST be specified
    as follows (no separators):

    address1ID (4 bytes) address2ID (4 bytes) Heuristics(1 byte)

    Output JSON file is dumped as visualize/graph.json to be viewed through visualize/index.html

    Returns void
    """
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

def get_data(data_src, percent_bytes=None):
    fn = "blockchain/data"
    if percent_bytes is not None:
        size_cmd = ["wget", "--spider", data_src]
        result = subprocess.run(size_cmd, stderr=subprocess.PIPE)
        bash_output = result.stderr.decode('utf-8')
        size_output = [line for line in bash_output.split("\n") if "length" in line.lower()][0]
        total_bytes = int(size.output.split(":")[1].trim())

        num_bytes = int(total_bytes * percent_bytes)

        download_command = "curl https://s3.amazonaws.com/bitcoinclustering/cluster_data.dat" \
            "| head -c {} > {}".format(num_bytes, fn)
        subprocess.call(download_command)        

    size = count_nodes(fn)
    return _create_similarity(fn, size)
        
if __name__ == "__main__":
    G = create_visual_json("blockchain/data")