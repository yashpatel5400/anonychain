"""
__author__ = Yash Patel
__name__   = metis.py
__description__ = Formats the raw blockchain graph data into the format required for
analysis by the METIS package
"""

import pickle

def format_metis(fn):
    print("Loading pickled data...")
    S = pickle.load(open(fn, "rb"))
    rows, cols = S.shape
    
    with open("blockchain/metis.graph", "w") as f:
        print("Reformatting pickled data...")
        for i in range(rows):
            cur_node = []
            for j in range(cols):
                if S[i,j] > 0: 
                    cur_node.append("{} {}".format(j+1, int(S[i,j])))
            f.write("{}\n".format(" ".join(cur_node)))

if __name__ == "__main__":
    format_metis("blockchain/data_0.000005.pickle")