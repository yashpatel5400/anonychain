Anonychain
=====================
Princeton Thesis 2017 (De-anonymization of Blockchain transcations)

Attempting to deanonymize wallets/addresses using a combination of ground truth labels of Bitcoin transactions and nodes, some similarity heuristics between nodes based on behavioral patterns, and clustering techniques (mostly spectral methods). Current deanonymization techniques mostly make use of union-find, which we wish to expand upon here.

Setup
=====================
All the setup is handled in downloading the appropriate _Python 3_ packages with:

```
sudo pip3 install -r requirements.txt
```

Documentation
=====================
There are two main modes of running the code: one for running a single SBM test (stochastic block model: [https://arxiv.org/pdf/1703.10146.pdf](https://arxiv.org/pdf/1703.10146.pdf)) and another for running many in succession to get aggregate accuracy results.

The SBM model generates a toy example based on parameters specified, which we then do spectral clustering on. The parameters to use are:

- (-r) : <run_test_bool>   (y/n) for whether to create SBM to run test or run on actual data
- (-d) : <display_bool>    (y/n) for whether to show PCA projections
- (-w) : <weighted_graph>  (y/n) for whether to have weights on edges (randomized)
- (-c) : <cluster_size>    (int) size of each cluster (assumed to be same for all)
- (-n) : <num_cluster>     (int) number of clusters (distinct people)
- (-g) : <guess_bool>      (y/n) to guess the number of clusters vs. take it as known
- (-p) : <p_value>         (0,1) for in-cluster probability
- (-q) : <q_value>         (0,1) for non-cluster probability
- (--cs) : <cluster_sizes> (int list) size of each cluster (comma delimited)
- (--lib) :                ('matplotlib','plotly') for plotting library

So, for example, the script would be executed with:

```
python3 app.py -g n -p .70 -q 0.10 --cs 7,10,8,4,6,10,3
```

This will produce output saying:

```
hierarchical accuracy: xyz%
k-means accuracy: xyz%
```

And graphs in the output folder. Of relevance are the eigen_guess.png, kmean_guess.png, and truth.png, which respectively represent the guessed clusters from the hierarchical implementation of spectral clustering, the k-means version, and the ground truth clusters. The accuracies printed in the console similarly correspond to the accuracies obtained in the clustering obtained through these two versions.

Running accuracy.py (without arguments) runs several trials on various p, q values and produces graphs plotting the relation in the output/accuracies folder. Namely, just call the script w/:

```
python3 accuracy.py
```