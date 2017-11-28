"""
__author__ = Yash Patel
__name__   = accuracy.py
__description__ = Runs tests on the spectral clustering deanonymization scripts
to see performance in hierarchical clustering vs. k-means. Does tests over
various values of p, q, and cluster sizes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sbm import create_sbm, create_clusters
from deanonymize import calc_accuracy, deanonymize
from constants import colors

def conduct_tests(ps, qs, css):
    trials = 5
    
    for cs in css:
        clusters = create_clusters(cs)

        for p in ps:
            hier_accuracies, kmeans_accuracies = [], []
            for q in qs:
                if q > p: break
                
                hier_trials, kmeans_trials = [], []
                for _ in range(trials):
                    sbm = create_sbm(clusters, p, q)
                    hier_partitions, kmeans_partitions = deanonymize(sbm, k=len(clusters))
                    hier_accuracy   = calc_accuracy(clusters, hier_partitions)
                    kmeans_accuracy = calc_accuracy(clusters, kmeans_partitions)

                    hier_trials.append(hier_accuracy)
                    kmeans_trials.append(kmeans_accuracy)

                hier_accuracies.append(np.mean(hier_trials))
                kmeans_accuracies.append(np.mean(kmeans_trials))

            print("Completed accuracy for: q={}, cs={}".format(q, cs))
            for accuracies, fn in zip([hier_accuracies, kmeans_accuracies],
                ["hierarchical.png", "kmeans.png"]):

                fig = plt.figure()
                plt.scatter(ps, accuracies, c=colors)
                
                plt.title("{} vs. p".format(fn))
                plt.xlabel("p")
                plt.ylabel("accuracy (%_correct)")

                plt.savefig("output/accuracy/q={}_cs={}_{}".format(q,cs,fn))
                plt.close()

def main():
    ps  = [i / 10 for i in range(1)]
    qs  = [i / 10 for i in range(1)]

    css = [
        [5,4,3,2]
    ]

    conduct_tests(ps, qs, css)
    
if __name__ == "__main__":
    main()