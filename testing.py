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
import KDE.standard_kde as standard_kde
import scipy.stats as stats
from sklearn.neighbors import KernelDensity

def silverman_scalar_bandwidth(training_data):
    n,d = training_data.shape
    out = 0
    const = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
    for i in range(d):
        out += const * np.std(training_data[:,i])
    return out / d

max_deg = 2

'''
n_vars = 10

n_samples = 1000
linear_gaussian_net = linear_gaussian_generation.generate_sparse_linear_gaussian_system(n_vars, max_deg, (0.2, 1), (-1, 1))

X = np.asarray(linear_gaussian_net.get_joint_samples(n_samples)).astype(np.float64)

#print("X: ", X)
'''
X = load_data.load_kde_cleaned_airline_data("Iberia").to_numpy()
#X = X[:1000, :]

X_train = X[:int(0.7 * X.shape[0]), :]
X_test = X[int(0.7 * X.shape[0]):, :]
bandwidth = silverman_scalar_bandwidth(X_train)
print("bandwidth: ", bandwidth)
kde_on_X = KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(X_train)
kde_on_X_test_log_likelihood = np.sum(kde_on_X.score_samples(X_test))
normal_dist_on_X = stats.multivariate_normal(mean = np.mean(X_train, axis = 0), cov = np.cov(X_train.T))
normal_dist_on_X_test_log_likelihood = np.sum(normal_dist_on_X.logpdf(X_test))
print("X shape: ", X.shape)


initial_dags = [random_graph.random_dag(X.shape[1], max_deg) for i in range(0, 1)]

kernel = 'gaussian'


print("kde_on_X_test_log_likelihood: ", kde_on_X_test_log_likelihood)
print("normal dist test log likelihood: ", normal_dist_on_X_test_log_likelihood)
#placeholders. Phillip, pls pick good values for these
initial_temp = kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood(X_train, X_test, kernel, 0, 0, negative = True)(initial_dags[0])
#generally seems to be the case that the baseline log likelihood as at the least half the intial temperature
final_temp = initial_temp / 2
alpha = (initial_temp - final_temp) / 5000
#simulated_annealing_modified(initial_state, initial_temp, final_temp, alpha, initial_cost, get_cost_ratio, get_neighbors_and_edge, print_iters = 50):
opt_dag = simulated_annealing.simulated_annealing_modified(initial_dags[0], \
    initial_temp,\
    final_temp,\
    alpha, \
    initial_temp,\
    kde_bayesian_net_log_likelihood.bayesian_net_log_likelihood_differential(X_train, X_test, kernel, 0, 0, negative = True),\
    simulated_annealing_dag.degree_constrained_neighbors_func_modified(max_deg),\
    print_iters = 10)

kde_on_X_test_log_likelihood = np.sum(np.log(kde_on_X.evaluate(X_test.T)))
