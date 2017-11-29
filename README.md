Anonychain
~~~~~~~~~~~~~~~~~~
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

- (-c)    : (int) size of each cluster (if you want a bunch of clusters of the same size, otherwise use the --cs option described below)
- (-n)    : (int) number of clusters (MUST be used in conjunction with -c, to specify the size of each of the clusters)
- (-g)    : (y/n) a boolean that indicates whether the number of clusters should be guess or taken as known
- (-p)    : [0,1] in-cluster probability
- (-q)    : [0,1] non-cluster probability
- (--cs)  : (int list) size of each cluster (comma delimited)
- (--lib) : ('matplotlib','plotly') for specifying which plotting library to use

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