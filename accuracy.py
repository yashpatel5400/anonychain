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

import plotly
import plotly.graph_objs as go

from setup.sbm import create_sbm, create_clusters
# from analysis.deanonymize import calc_accuracy, deanonymize

def extract_results(p, qs):
    alg_accuracies, alg_times = {}, {} 
    for q in qs:
        fn = "output/results_p-{}_q-{}.txt".format(p, q)
        lines = open(fn, "r").readlines()
        accuracies, times = lines[3:9], lines[13:19]
        for accuracy in accuracies:
            values = accuracy.split("|")
            alg_name, alg_accuracy = values[1].strip(), float(values[2].strip())
            if alg_name not in alg_accuracies:
                alg_accuracies[alg_name] = []
            alg_accuracies[alg_name].append(alg_accuracy)

        for time in times:
            values = time.split("|")
            alg_name, alg_time = values[1].strip(), float(values[2].strip())
            if alg_name not in alg_times:
                alg_times[alg_name] = []
            alg_times[alg_name].append(alg_time)

    return alg_accuracies, alg_times

def graph_results(fn, d, p, qs):
    data = []
    for alg in d:
        data.append(go.Scatter(
            x=qs,
            y=d[alg],
            name=alg
        ))

    layout = dict(
        xaxis = dict(title = 'qs'),
        yaxis = dict(title = fn),
    )
    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename='output/results_{}-{}.html'.format(fn, p))

def conduct_tests(ps, qs, css):
    """Given lists of p probabilities, q probabilities, and lists of cluster sizes,
    runs tests on clustering accuracies (both hierarchical and k-means)

    Returns void
    """
    trials = 5
    
    for cs in css:
        clusters = create_clusters(cs)

        for p in ps:
            hier_accuracies, kmeans_accuracies = [], []
            for i, q in enumerate(qs):
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

            print("Completed accuracy for: p={}, cs={}".format(p, cs))
            for accuracies, label in zip([hier_accuracies, kmeans_accuracies],
                ["hierarchical", "kmeans"]):

                fig = plt.figure()
                plt.scatter(qs[:i], accuracies)
                
                plt.title("{} vs. q (p={}_cs={})".format(label, p,cs))
                plt.xlabel("q")
                plt.ylabel("accuracy (%_correct)")

                plt.savefig("output/accuracy/p={}_cs={}_{}.png".format(p, cs, label))
                plt.close()

def main():
    ps = [0.6, 0.7, 0.8, 0.9, 1.0]
    qs = [0.0, 0.1, 0.2]
    for p in ps:
        accuracies, times = extract_results(p, qs)
        graph_results("accuracies", accuracies, p, qs)
        graph_results("times", times, p, qs)
        print("Finished graphing p={}".format(p))

if __name__ == "__main__":
    main()