"""
__author__ = Yash Patel
__name__   = metis.py
__description__ = Formats the raw blockchain graph data into the format required for
analysis by the METIS package
"""

import subprocess
import pickle
import networkx as nx
from scipy.sparse import dok_matrix, csr_matrix
import matplotlib
matplotlib.use('Agg')

from analysis.deanonymize import draw_results

def format_metis(S, metis_fn):
    rows, cols = S.shape
    if   isinstance(S, dok_matrix):
        edges = dok_matrix.count_nonzero(S) // 2
    elif isinstance(S, csr_matrix):
        edges = csr_matrix.count_nonzero(S) // 2

    with open(metis_fn, "w") as f:
        # header format: #nodes, #edges, 001 (indicates weighted edges)
        f.write("{} {} 001\n".format(rows, edges))
        for i in range(rows):
            cur_node = []
            for j in range(cols):
                if S[i,j] > 0:
                    cur_node.append("{} {}".format(j+1, int(S[i,j])))
            f.write("{}\n".format(" ".join(cur_node)))

def run_metis(metis_fn, num_partitions):
    pmetis_cmd = ["pmetis", metis_fn, str(num_partitions)]
    result = subprocess.run(pmetis_cmd, stdout=subprocess.PIPE)

    output_lines = result.stdout.decode("utf8").split("\n")
    time_elapsed = float([line for line in output_lines 
        if "Total:" in line][0].split("Total:")[1])

    result_fn = "{}.part.{}".format(metis_fn, num_partitions)
    partitions = [set() for _ in range(num_partitions)]
    with open(result_fn) as f:
        line_no = 0
        for line in f:
            partitions[int(line.strip())].add(line_no)
            line_no += 1
    return partitions, time_elapsed

def metis_from_pickle(fn, num_partitions):
    print("Loading pickled data...")
    S = pickle.load(open("blockchain/{}.pickle".format(fn), "rb"))
    metis_fn = "blockchain/{}.graph".format(fn)
    
    print("Reformatting pickled data...")
    format_metis(S, metis_fn)
    return run_metis(metis_fn, num_partitions)

if __name__ == "__main__":
    metis_from_pickle("data_0.000050", num_partitions=250)
    # run_metis("blockchain/data_0.000050.graph", num_partitions=250)