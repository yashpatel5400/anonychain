"""
__author__ = Yash Patel
__name__   = accuracy.py
__description__ = Runs tests on the spectral clustering deanonymization scripts
to see performance in hierarchical clustering vs. k-means. Does tests over
various values of p, q, and cluster sizes
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
from collections import defaultdict

from PIL import Image, ImageFont, ImageDraw, ImageOps
from setup.sbm import create_sbm, create_clusters
# from analysis.deanonymize import calc_accuracy, deanonymize

fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')

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

def extract_results(accuracy_metric, p, qs):
    alg_accuracies, alg_times = {}, {} 
    for q in qs:
        fn = "output/{}_p-{}_q-{}.txt".format(accuracy_metric, p, q)
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

def graph_results(fn, d, p, qs, n, gc=None, save_static=True):
    xaxis = "qs"
    yaxis = fn
    if yaxis == "times":
        yaxis = "time (s)"

    if p is not None:
        title = "{} vs. {} for p={}".format(xaxis, yaxis, p)
        if gc is not None:
            out_fn = 'output/results_{}-{}-{}-{}.png'.format(fn, p, gc, n)
        else:
            out_fn = 'output/results_{}-{}-{}.png'.format(fn, p, n)
    else:
        title = "{} vs. {}".format(xaxis, yaxis)
        out_fn = 'output/results_{}.png'.format(fn)

    if save_static:
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(title)

        ax = fig.add_subplot(111)
        for alg in d:
            plt.plot(qs, d[alg], marker='o', label=alg)
        
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
          fancybox=True, shadow=True, ncol=6)

        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)

        plt.savefig(out_fn)
        plt.close()
    
    else:
        data = []
        for alg in d:
            data.append(go.Scatter(
                x=qs,
                y=d[alg],
                name=alg
            ))

        layout = dict(
            title = title,
            xaxis = dict(title = xaxis),
            yaxis = dict(title = yaxis),
        )
        fig = dict(data=data, layout=layout)
        plotly.offline.plot(fig, filename=out_fn)
    return out_fn

def create_collage(width, height, listofimages, output_fn):
    cols = 2
    rows = 3
    thumbnail_width = width//cols
    thumbnail_height = height//rows
    size = thumbnail_width, thumbnail_height
    new_im = Image.new('RGB', (width, height))
    ims = []
    for p in listofimages:
        im = Image.open(p)
        
        # draw = ImageDraw.Draw(im)
        # font = ImageFont.truetype(os.path.join(fonts_path, 'Archivo.ttf'), 16)
        # 
        # title = p.split("/")[-1].split("_")[0]
        # draw.text((50, 50),title,(0,0,0),font=font)
        
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            print(i, x, y)
            new_im.paste(ims[i], (x, y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0

    new_im.save("output/results_{}.png".format(output_fn))

def main(opt_params):
    files = os.listdir("output")
    accuracy_metrics = ["purity", "nmi", "weighted_ri"]

    alg_scores = {}
    avg_alg_scores = {}
    accuracy_weights = [1,3,2]
    total_weight = sum(accuracy_weights)

    for i, accuracy_metric in enumerate(accuracy_metrics):
        results = [file for file in files if len(file.split(accuracy_metric)) > 1
            and file.split(".")[-1] == "txt" and opt_params in file]
        p_to_qs = {}
        for result in results:
            params_txt = result.split(accuracy_metric)[1]
            params = params_txt.split("_")
            p = float(params[1].split("-")[1])
            q = float(params[2].split("-")[1])

            gc = None
            if "gc" in params_txt:
                gc = int(params[3].split("-")[1])
            n = int(params[-1].split("-")[1].split(".txt")[0])

            if p not in p_to_qs:
                p_to_qs[p] = []
            p_to_qs[p].append(q)

        result_files = []
        for p in p_to_qs:
            sorted_qs = sorted(p_to_qs[p])
            accuracies, times = extract_results(accuracy_metric, p, sorted_qs)
            for alg in accuracies:
                if p not in alg_scores:
                    alg_scores[p] = {}
                if alg not in alg_scores[p]:
                    alg_scores[p][alg] = [0] * len(accuracies[alg])
                for j, accuracy in enumerate(accuracies[alg]):
                    alg_scores[p][alg][j] += accuracy_weights[i] * accuracy / total_weight
                    
            out_result = graph_results(accuracy_metric, accuracies, p, sorted_qs, n, gc)
            result_files.append(out_result)
            print("Finished graphing {} for p={}".format(accuracy_metric, p))
        # create_collage(1200 * 2,800 * 3,result_files, accuracy_metric)
    
    # result_files = []
    # sorted_ps = sorted(alg_scores.keys())
    # for j, p in enumerate(sorted_ps):
    #     out_result = graph_results("overall", alg_scores[p], p, p_to_qs[p], n)
    #     result_files.append(out_result)
    #     
    #     for alg in alg_scores[p]:
    #         if alg not in avg_alg_scores:
    #             avg_alg_scores[alg] = [0] * len(sorted_ps)
    #         avg_alg_scores[alg][j] = np.mean(alg_scores[p][alg])
    #         
    # graph_results("average", avg_alg_scores, None, sorted_ps, n)
    # create_collage(1200 * 2,800 * 3,result_files, "overall")

if __name__ == "__main__":
    main(opt_params="gc-1_n-276")
    # files = [
    #     "output/p-0.9_q-0.15/KMeans_p-0.9_q-0.15.png",
    #     "output/p-0.9_q-0.15/ManualHierarchical_p-0.9_q-0.15.png",
    #     "output/p-0.9_q-0.15/ManualKmeans_p-0.9_q-0.15.png",
    #     "output/p-0.9_q-0.15/Metis_p-0.9_q-0.15.png",
    #     "output/p-0.9_q-0.15/MiniBatchKMeans_p-0.9_q-0.15.png",
    #     "output/p-0.9_q-0.15/SpectralClustering_p-0.9_q-0.15.png"
    # ]
    # create_collage(650 * 2,500 * 3,files, "p-0.9_q-0.15")