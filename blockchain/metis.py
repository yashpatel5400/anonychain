"""
__author__ = Yash Patel
__name__   = metis.py
__description__ = Formats the raw blockchain graph data into the format required for
analysis by the METIS package
"""

import subprocess
import pickle
import networkx as nx
from scipy.sparse import dok_matrix
import matplotlib
matplotlib.use('Agg')

from analysis.deanonymize import draw_results

def format_metis(S, metis_fn):
    rows, cols = S.shape
    edges = dok_matrix.count_nonzero(S) // 2
    with open(metis_fn, "w") as f:
        # header format: #nodes, #edges, 001 (indicates weighted edges)
        f.write("{} {} 001\n".format(rows, edges))
        for i in range(rows):
            cur_node = []
            for j in range(cols):
                if S[i,j] > 0: 
                    cur_node.append("{} {}".format(j+1, int(S[i,j])))
            f.write("{}\n".format(" ".join(cur_node)))

def metis_partition(fn, num_partitions):
    print("Loading pickled data...")
    S = pickle.load(open("blockchain/{}.pickle".format(fn), "rb"))
    metis_fn = "blockchain/{}.graph".format(fn)
    
    print("Reformatting pickled data...")
    format_metis(S, metis_fn)

    pmetis_cmd = ["pmetis", metis_fn, str(num_partitions)]
    result = subprocess.run(pmetis_cmd)

    result_fn = "{}.part.{}".format(metis_fn, num_partitions)
    partitions = [set() for _ in range(num_partitions)]
    with open(result_fn) as f:
        line_no = 0
        for line in f:
            partitions[int(line.strip())].add(line_no)
            line_no += 1

    G = nx.from_scipy_sparse_matrix(S)
    spring_pos = nx.spring_layout(G)
    draw_results(G, spring_pos, partitions, "metis_guess.png")

if __name__ == "__main__":
    metis_partition("data_0.000000", num_partitions=25)