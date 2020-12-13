from rand import random_graph as random_graph
import numpy as np
from bayesian_net.MontyHallBayesNet import MontyBayesNet
from rand.sampling import bayesian_net_sampler
from bayesian_net.linear_gaussian_bayesian_net import LinearGaussianBayesianNet
import optimization.genetic_optimizer_dag as genetic_optimizer_dag
import optimization.genetic_optimizer as genetic_optimizer
import rand.synthetic.linear_gaussian_generation as linear_gaussian_generation
import optimization.objectives.kde_bayesian_net_log_likelihood as kde_bayesian_net_log_likelihood
import KDE.KDE as KDE

n_vars = 7
max_deg = 3
n_samples = 500
linear_gaussian_net = linear_gaussian_generation.generate_sparse_linear_gaussian_system(n_vars, max_deg, (0.2, 1), (-1, 1))

X = np.asarray(linear_gaussian_net.get_joint_samples(n_samples)).astype(np.float64)
X_train = X[:int(0.7 * X.shape[0]), :]
X_test = X[int(0.7 * X.shape[0]):, :]

#print("X: ", X)

initial_dags = [random_graph.random_dag(n_vars, max_deg) for i in range(0, 200)]

kernel = 'gaussian'#KDE.guassian_kernel

opt_dags = genetic_optimizer.optimize(initial_dags, \
    kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood(X_train, X_test, kernel, 1, 1),
    15,  \
    genetic_optimizer.proportional_product_selection_probability_func,
    genetic_optimizer_dag.dag_crossover_fast,
    genetic_optimizer_dag.mutate_dag_func(0, 10),\
    10, \
    print_iters = 1)
print("opt_dag: \n", opt_dags[0])
print("correct dag: \n", linear_gaussian_net.get_dag())
print("opt vs. correct: ", np.average(opt_dags[0] == linear_gaussian_net.get_dag()))


'''
#dag merging testing
n = 100
min_deg = int(0.3*n)
max_deg = int(0.6*n)
for i in range(0, 1000):
    print("i: ", i)
    A = random_graph.random_dag(n, np.random.randint(min_deg, high = max_deg))
    B = random_graph.random_dag(n, np.random.randint(min_deg, high = max_deg))
    N = genetic_optimizer_dag.dag_crossover_fast(A, B)
'''
