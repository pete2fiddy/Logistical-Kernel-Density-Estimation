from rand import random_graph as random_graph
import numpy as np
import rand.random_graph
from bayesian_net.MontyHallBayesNet import MontyBayesNet
from rand.sampling import bayesian_net_sampler
from bayesian_net.linear_gaussian_bayesian_net import LinearGaussianBayesianNet



monty_bayes_net = MontyBayesNet()
#Guest picks A, Prize was under B, Monty shows A, Not possible
print(monty_bayes_net.joint_prob(np.array(['A','B','A'])))
#Guest picks A, Prize was under A, Monty will show either B or C
print(monty_bayes_net.joint_prob(np.array(['A','A','B'])))
#Guest picks A, Prize was under B, Monty will always show C
print(monty_bayes_net.joint_prob(np.array(['A','B','C'])))
monty_bayes_net.predict_prob(np.array(['A',None,'B']),1,['A','B','C'])

samples = np.asarray(monty_bayes_net.get_joint_samples(50)).astype(np.str)
print("samples: \n", samples)



linear_gaussian_dag = random_graph.random_dag(15, 8)
print("linear gaussian dag: \n", linear_gaussian_dag)

W = 2 * (np.random.rand(linear_gaussian_dag.shape[0], linear_gaussian_dag.shape[1]) - 0.5) * linear_gaussian_dag
biases = np.random.rand(linear_gaussian_dag.shape[0])
std_devs = 0.2 + np.random.rand(linear_gaussian_dag.shape[0])

linear_gaussian_net = LinearGaussianBayesianNet(W, biases, std_devs)
linear_gaussian_samples = np.asarray(linear_gaussian_net.get_joint_samples(50)).astype(np.float64)
print("linear gaussian samples: \n", linear_gaussian_samples)
