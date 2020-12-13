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
import optimization.simulated_annealing as simulated_annealing
import optimization.simulated_annealing_dag as simulated_annealing_dag
import KDE.load_data as load_data
'''
n_vars = 10
max_deg = 5
n_samples = 1000
linear_gaussian_net = linear_gaussian_generation.generate_sparse_linear_gaussian_system(n_vars, max_deg, (0.2, 1), (-1, 1))

X = np.asarray(linear_gaussian_net.get_joint_samples(n_samples)).astype(np.float64)
X_train = X[:int(0.7 * X.shape[0]), :]
X_test = X[int(0.7 * X.shape[0]):, :]

#print("X: ", X)

initial_dags = [random_graph.random_dag(n_vars, max_deg) for i in range(0, 1)]

kernel = 'gaussian'


#placeholders. Phillip, pls pick good values for these
initial_temp = kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood(X_train, X_test, kernel, 0, 0, negative = True)(initial_dags[0])
#generally seems to be the case that the baseline log likelihood as at the least half the intial temperature
final_temp = initial_temp / 2
alpha = (initial_temp - final_temp) / 5000
print("linear gaussian log likelihood: ", np.sum(np.log(linear_gaussian_net.joint_prob(X_test))))
opt_dag = simulated_annealing.simulated_annealing(initial_dags[0], \
    initial_temp,\
    final_temp,\
    alpha, \
    kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood(X_train, X_test, kernel, 0, 0, negative = True),\
    simulated_annealing_dag.degree_constrained_neighbors_func(max_deg),\
    print_iters = 10)

print("linear gaussian log likelihood: ", np.sum(np.log(linear_gaussian_net.joint_prob(X_test))))
'''

X = load_data.load_kde_cleaned_airline_data("Iberia")
print("X: \n", X)
