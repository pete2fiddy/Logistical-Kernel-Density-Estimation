import numpy as np


'''
returns: a function which takes in a DAG, dag, and returns the log-likelihood
of the dataset X (with rows that are data points) according to the bayesian network
given by net_initializer_func(dag).
'''
def bayesian_net_log_likelihood(X, net_initializer_func):
    #There is a possibility that a log-sum-exp trick may need to be used
    #in order to prevent numerical errors within the joint probability calculations
    #in the bayesian network. if this is the case, we should also add a "log_joint_prob"
    #function to the BayesianNetwork implementation.
    def out(dag):
        bayesian_net = net_initializer_func(dag)
        probs = np.zeros(X.shape[0], dtype = np.float64)
        for i in range(X.shape[0]):
            probs[i] = bayesian_net.joint_prob(X[i])
        return np.sum(np.log(probs))

    return out
