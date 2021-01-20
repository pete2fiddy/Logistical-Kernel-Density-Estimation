## Goal

Kernel density estimators are a form of nearest-neighbor joint density estimate. They tend to suffer in high dimensions, as most nearest-neighbor approaches do, as points are nearly always very far apart by Euclidean distance. We seek to remedy this issue by leveraging the assumption that the underlying joint distribution is "conditionally sparse" -- that is, the features of the data are conditionally dependent upon only a small number of the other features.

## Bayesian Networks

A Bayesian network is a form of probabilistic graphical model that encodes the conditional dependencies of the joint distribution. Given a directed acyclic graph (DAG), with a node for each random variable, the parents of each node are the random variables upon which that random variable is conditionally dependent. Through repeated application of the product rule of probability, eliminating the conditionally independent terms of each conditional factor, the following joint probability distribution is derived from the graph structure:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\prod_{i=1}^{d}&space;p(x_i&space;|&space;\text{pa}(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\prod_{i=1}^{d}&space;p(x_i&space;|&space;\text{pa}(x_i))" title="p(x) = \prod_{i=1}^{d} p(x_i | \text{pa}(x_i))" /></a>

Where pa(x[i]) is the subset of elements of x formed from the parents of node i in the graph.

## Kernel Density Estimators

A kernel density estimator (KDE) is a nearest-neighbor joint density estimate of the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\frac{1}{N}\sum_{i&space;=&space;1}^{N}&space;k(x&space;-&space;x_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\frac{1}{N}\sum_{i&space;=&space;1}^{N}&space;k(x&space;-&space;x_i)" title="p(x) = \frac{1}{N}\sum_{i = 1}^{N} k(x - x_i)" /></a>

Where k is a valid kernel function that integrates to 1. A popular kernel for kernel density estimation is the Gaussian kernel of the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=k(x)&space;=&space;\frac{1}{\sqrt{2\pi&space;h^2}}&space;e^{-\frac{\|x\|^2}{2h^2}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k(x)&space;=&space;\frac{1}{\sqrt{2\pi&space;h^2}}&space;e^{-\frac{\|x\|^2}{2h^2}}" title="k(x) = \frac{1}{\sqrt{2\pi h^2}} e^{-\frac{\|x\|^2}{2h^2}}" /></a>

Where h is a bandwidth parameter that controls the extent to which more distant points influence the density estimate.

KDEs tend to suffer in high dimensions due to the curse of dimensionality. Primarily, they suffer because as the number of dimensions increase, the Euclidean norm of an arbitrary point grows significantly. This results in kernel outputs in the KDE that are incredibly small, or even zero, if a kernel with finite support is used.


## Our Approach

Recall that the joint distribution derived from a Bayesian network is of the form:


<a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\prod_{i=1}^{d}&space;p(x_i&space;|&space;\text{pa}(x_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\prod_{i=1}^{d}&space;p(x_i&space;|&space;\text{pa}(x_i))" title="p(x) = \prod_{i=1}^{d} p(x_i | \text{pa}(x_i))" /></a>

Now, imagine that each conditional probability of the product were represented as follows:

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x_i|\text{pa}(x_i))&space;=&space;\frac{p(x_i,&space;\text{pa}(x_i))}{p(\text{pa}(x_i))}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x_i|\text{pa}(x_i))&space;=&space;\frac{p(x_i,&space;\text{pa}(x_i))}{p(\text{pa}(x_i))}" title="p(x_i|\text{pa}(x_i)) = \frac{p(x_i, \text{pa}(x_i))}{p(\text{pa}(x_i))}" /></a>

Where both the numerator and denominators are modeled by kernel density estimators. Assuming the number of parents each node has in the DAG is small, then these KDEs will be of low dimension, and hence will not suffer from the curse of dimensionality. We henceforth refer to such a Bayesian network with small sets of parents for each node as "conditionally sparse."


## Optimization Procedure

While we have identified a solution for the problems KDEs face in high dimensions, the structure of the underlying Bayesian network is still unknown. We apply discrete optimization algorithms to determine a high-quality conditionally sparse Bayesian network for the data when each node's conditional probability distribution is constructed using kernel density estimators. This is accomplished through maximizing the log-likelihood of a "held-out" dataset that was not used to train the KDEs of the system. Conditional sparsity is enforced by requiring that the size of each set of parent nodes is at most some constant chosen by the user. This condition is particularly straightforward to enforce using the simulated annealing algorithm by only considering neighboring DAGs that meet this criteria, which was the reason we chose to use it.

## Relation to Logistical Tasks

In logistical tasks, minimizing the number of interactions between components prevents a complicated mess of dependent pieces from forming. This increases the interpretability and flexibility of the system and mitigates the risk of catastrophic failure due to the malfunction of a highly dependent component. As such, logistical systems are often explicitly formulated such that its components are as independent as possible, and as a result, logistical densities tend to be conditionally sparse. As a result, Bayesian networks are strong candidates for modeling logistical densities.

However, the structure of the network isn't always obvious, especially in larger logistical systems. Our method determines a strong graphical structure for the system by leveraging the conditional sparsity properties logistical tasks tend to display, and results in a joint density constructed to explicitly leverage this property. As such, this approach appears especially well-suited to determining the conditional dependence structure of logistical systems and their underlying probability densities.
