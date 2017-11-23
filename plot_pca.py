"""
__author__ = Yash Patel
__name__   = plot_pca.py
__description__ = Functions to plot PCA results (for 2, 3 dimensions)
"""

from sklearn.decomposition import PCA
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly
from plotly.graph_objs import Scatter, Scatter3d, Layout

def _plot_clusters_2d(colors, A, plot_lib):
    pca = PCA(n_components=2)
    pca_A = pca.fit_transform(A)

    pca_x, pca_y = [], []
    for pca_val in pca_A:
        pca_x.append(pca_val[0])
        pca_y.append(pca_val[1])

    if plot_lib == "matplotlib":
        plt.scatter(pca_x, pca_y, c=colors)
        plt.savefig("output/pca_2d.png")
        plt.close()
    elif plot_lib == "plotly":
        plotly.offline.plot({
            "data": [Scatter(
                x=pca_x, 
                y=pca_y, 
                mode='markers', 
                marker = dict(color=colors)
            )],
            "layout": Layout(title="PCA 2D Projection")
        },  filename="output/pca_2d.html")

def _plot_clusters_3d(colors, A, plot_lib):
    pca = PCA(n_components=3)
    pca_A = pca.fit_transform(A)

    pca_x, pca_y, pca_z = [], [], []
    for pca_val in pca_A:
        pca_x.append(pca_val[0])
        pca_y.append(pca_val[1])
        pca_z.append(pca_val[2])

    if plot_lib == "matplotlib":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_x, pca_y, pca_z, c=colors)
        plt.savefig("output/pca_3d.png")
        plt.close()
    elif plot_lib == "plotly":
        plotly.offline.plot({
            "data": [Scatter3d(
                x=pca_x, 
                y=pca_y, 
                z=pca_z, 
                mode='markers', 
                marker = dict(color=colors)
            )],
            "layout": Layout(title="PCA 3D Projection")
        },  filename="output/pca_3d.html")

def plot_pca(G, clusters, plot_2d=True, plot_3d=True, plot_lib="plotly"):
    A = nx.to_numpy_matrix(G)
    colors = ["blue", "green", "red", "cyan", "black", "pink"]
    scatter_colors = [colors[i] for i in range(len(clusters)) for _ in clusters[i]]

    if plot_2d:
        _plot_clusters_2d(scatter_colors, A, plot_lib=plot_lib)
    if plot_3d:
        _plot_clusters_3d(scatter_colors, A, plot_lib=plot_lib)