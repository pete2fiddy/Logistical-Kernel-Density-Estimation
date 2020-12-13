import numpy as np
from bayesian_net.linear_gaussian_bayesian_net import LinearGaussianBayesianNet
from rand import random_graph as random_graph

def generate_sparse_linear_gaussian_system(dim, max_deg, std_dev_bounds, bias_bounds):
    linear_gaussian_dag = random_graph.random_dag(dim, max_deg)

    W = 2 * (np.random.rand(linear_gaussian_dag.shape[0], linear_gaussian_dag.shape[1]) - 0.5) * linear_gaussian_dag
    biases = bias_bounds[0] + (bias_bounds[1] - bias_bounds[0]) * np.random.rand(linear_gaussian_dag.shape[0])
    std_devs = std_dev_bounds[0] + (std_dev_bounds[1] - std_dev_bounds[0]) * np.random.rand(linear_gaussian_dag.shape[0])

    linear_gaussian_net = LinearGaussianBayesianNet(W, biases, std_devs)

    return linear_gaussian_net
