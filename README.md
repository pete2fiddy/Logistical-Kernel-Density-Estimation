Kernel density estimators are a form of nearest-neighbor joint density estimate. They tend to suffer, as most nearest-neighbor approaches do, in high dimensions as points are nearly always very far apart by L2 distance. We seek to remedy this issue by leveraging the assumption that the underlying joint distribution is "conditionally sparse" -- that is, the features of the data are conditionally dependent upon only a small number of the other features.

## Bayesian Networks

A Bayesian Network is a form of probabilistic graphical model that encodes the conditional dependencies of the joint distribution. Given a directed acyclic graph (DAG), with a node for each random variable, the parents of each node are the random variables upon which that random variable is conditionally dependent. Through repeated application of the product rule of probability, eliminating the conditionally independent terms of each conditional factor, the following joint probability distribution is derived from the graph structure:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\prod_{i=1}^{d}&space;p(x_i&space;|&space;\text{pa}(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\prod_{i=1}^{d}&space;p(x_i&space;|&space;\text{pa}(x_i))" title="p(x) = \prod_{i=1}^{d} p(x_i | \text{pa}(x_i))" /></a>

Where pa(x[i]) is the subset of elements of x formed from the parents of node i in the graph.

## Kernel Density Estimators

## Our Approach

## Optimization Procedure
