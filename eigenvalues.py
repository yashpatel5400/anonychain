import random
import networkx as nx
import numpy as np

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def create_clusters(cluster_sizes):
    completed_nodes = 0
    clusters = []
    for cluster_size in cluster_sizes:
        clusters.append(set(range(completed_nodes, completed_nodes + cluster_size)))
        completed_nodes += cluster_size
    return clusters

def create_sbm(cluster_sizes, p, q):
    """
    creates a stochastic block model, with probability p within the same
    cluster and probability q outside, with n vertices
    """
    G = nx.Graph()
    total_nodes = sum(cluster_sizes)
    G.add_nodes_from(list(range(total_nodes)))

    clusters = create_clusters(cluster_sizes)
    verify_same = 0
    verify_diff = 0

    same_cluster_nodes = 0
    diff_cluster_nodes = 0
    for cluster in clusters:
        for cur_node in cluster:
            for other_node in range(total_nodes):
                if other_node == cur_node:
                    continue
                if other_node in cluster:
                    prob = p
                    same_cluster_nodes += 1
                else: 
                    prob = q
                    diff_cluster_nodes += 1                    

                if random.random() < prob:
                    if other_node in cluster:
                        verify_same += 1
                    else: verify_diff += 1
                    G.add_edge(cur_node, other_node)

    print("Prop same: {}; Prop diff: {}".format(
        verify_same/same_cluster_nodes, verify_diff/diff_cluster_nodes))
    return G

def plot_clusters(n_components, colors, A):
    pca = PCA(n_components=n_components)
    pca.fit(A)
    pca_A = pca.transform(A)

    pca_x = [pca_val[0] for pca_val in pca_A]
    pca_y = [pca_val[1] for pca_val in pca_A]

    plt.scatter(pca_x, pca_y, c=colors)
    plt.savefig("pca.png")

def get_eigenvectors(A):
    U, s, V = np.linalg.svd(A)

    rep_eigenvectors = U[:2]
    min_distances = []
    for eigenvector in U:
        distances = [np.linalg.norm(eigenvector - rep_eigenvector) for 
            rep_eigenvector in rep_eigenvectors]
        min_distances.append(min(distances))

    distinct_jumps = []
    for i in range(1, len(s)):
        if (s[i-1] - s[i])/s[i-1] > MARGIN:
            distinct_jumps.append(s[i-1])
    # print(distinct_jumps)

def main():
    MARGIN = .25

    p = .75
    q = .25
    cluster_sizes = [10,10]
    
    sbm = create_sbm(cluster_sizes, p, q)
    A = nx.to_numpy_matrix(sbm)

    n_components = 2
    colors = ["r","b"]
    scatter_colors = [colors[i] for i in range(len(cluster_sizes)) 
        for _ in range(cluster_sizes[i])]
    plot_clusters(n_components, scatter_colors, A)

if __name__ == "__main__":
    main()