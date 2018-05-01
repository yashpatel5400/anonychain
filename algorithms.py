"""
__author__ = Yash Patel
__name__   = algorithms.py
__description__ = Defines the clustering algorithms to be used and tested for the
deanonymization graph data
"""

import sklearn.cluster

def get_algorithms(num_clusters):
    return {
        "KMeans": (sklearn.cluster.KMeans, (), {
            'n_clusters' : num_clusters,
            'n_jobs' : -1,
            'algorithm' : 'full'
        }),
        
        "MiniBatchKMeans": (sklearn.cluster.MiniBatchKMeans, (), {
            'n_clusters' : num_clusters
        }),
           
        "SpectralClustering": (sklearn.cluster.SpectralClustering, (), {
            'n_clusters' : num_clusters,
            'n_jobs' : -1,
            'affinity' : 'rbf',
            'assign_labels' : 'discretize'
        }),
        
        "AffinityPropagation": (sklearn.cluster.AffinityPropagation, (), {
            'damping' : .5,
            'affinity' : 'euclidean'
        }),
        
        "DBSCAN": (sklearn.cluster.DBSCAN, (), {
            'eps' : 10.0,
            'n_jobs' : -1
        })
    }