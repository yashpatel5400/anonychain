Sparsify
=====================
Sparsifiers are implemented here, intended to remove as large a percent of the edges of
the input graph as possible while retaining some semblance of the original in some
capacity. This "capacity" is a metric that differs from sparsifier to sparsifier, i.e.

- Spectral sparsifier: Attempts to produce a graph that has a graph Laplacian as similar
to that of the original as possible: [Background](https://www.cs.ubc.ca/~nickhar/Cargese3.pdf)