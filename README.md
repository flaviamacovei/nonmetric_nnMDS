# nnMDS
Metric Multidimensional Scaling for Large Single-Cell Data Sets using Neural Networks 

Metric multidimensional scaling is one of the classical methods for embedding data into low-dimensional Euclidean space.
It creates the low-dimensional embedding by approximately preserving the pairwise distances between the input points. However,
current state-of-the-art approaches only scale to a few thousand data points. For larger data sets such as those occurring in single-cell
RNA sequencing experiments, the running time becomes prohibitively large and thus alternative methods such as PCA are widely
used instead. Here, we propose a neural network based approach for solving the metric multidimensional scaling problem that is
orders of magnitude faster than previous state-of-the-art approaches, and hence scales to data sets with up to a few million cells. At
the same time, it provides a non-linear mapping between high- and low-dimensional space that can place previously unseen cells in
the same embedding.
