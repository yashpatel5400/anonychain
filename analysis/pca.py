"""
__author__ = Yash Patel
__name__   = pca.py
__description__ = Functions to plot PCA results (for 2, 3 dimensions)
"""

from sklearn.decomposition import PCA
import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import plotly
from plotly.graph_objs import Scatter, Scatter3d, Layout

from analysis.constants import colors

def _plot_2d_pca(colors, A, plot_lib):
    """Given list of possible colors, the matrix representation of a graph (i.e.
    the adjacency matrix), and a plotting library (either 'matplotlib' or 'plotly'),
    plots PCA on 2 dimensions

    Returns void
    """
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
        },  filename="output/pca/2d.html")

def _plot_3d_pca(colors, A, plot_lib):
    """Given list of possible colors, the matrix representation of a graph (i.e.
    the adjacency matrix), and a plotting library (either 'matplotlib' or 'plotly'),
    plots PCA on 3 dimensions

    Returns void
    """
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
        },  filename="output/pca/3d.html")

def plot_pca(G, clusters, plot_2d=True, plot_3d=True, plot_lib="plotly"):
    """Given a graph G, the node clusters, whether a 2D PCA plot is desired, whether
    a 3D PCA plot is desired, and a plotting library (either 'matplotlib' or 'plotly'),
    plots the corressponding PCAs and saves in the output/pca/ folder 

    Returns void
    """
    A = nx.to_numpy_matrix(G)
    scatter_colors = [colors[i] for i in range(len(clusters)) for _ in clusters[i]]

    if plot_2d:
        _plot_2d_pca(scatter_colors, A, plot_lib=plot_lib)
    if plot_3d:
        _plot_3d_pca(scatter_colors, A, plot_lib=plot_lib)