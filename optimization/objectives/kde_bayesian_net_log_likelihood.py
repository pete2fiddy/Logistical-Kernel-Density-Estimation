import numpy as np
from bayesian_net.KDE_bayesian_network import KDEBayesianNetwork

'''
returns: a function which takes in a list of DAGs, dags, and returns the log-likelihoods
of the dataset X_test (with rows that are data points) according to the KDE
bayesian network initialized with training data X_train and kernel kernel.
'''
def bayesian_net_log_likelihood(X_train, X_test, kernel, lambd, regularizer_order, negative = False):
    #There is a possibility that a log-sum-exp trick may need to be used
    #in order to prevent numerical errors within the joint probability calculations
    #in the bayesian network. if this is the case, we should also add a "log_joint_prob"
    #function to the BayesianNetwork implementation.
    def out(dags):
        wasnt_list = False
        if not isinstance(dags, list):
            dags = [dags]
            wasnt_list = True
        likelihoods = np.zeros(len(dags), dtype = np.float64)
        for i in range(likelihoods.shape[0]):
            dag = dags[i]
            bayesian_net = KDEBayesianNetwork(dag, X_train, kernel)
            probs = bayesian_net.joint_prob(X_test)
            likelihoods[i] = np.sum(np.log(probs)) - lambd * np.sum(dag)**regularizer_order

        if negative:
            likelihoods *= -1
        
        if wasnt_list:
            return likelihoods[0]

        return likelihoods
        '''
        likelihoods = np.zeros(len(dags), dtype = np.float64)
        for i in range(likelihoods.shape[0]):
            dag = dags[i]
            bayesian_net = KDEBayesianNetwork(dag, X_train, kernel)
            probs = np.zeros(X_test.shape[0], dtype = np.float64)
            for j in range(X_test.shape[0]):
                probs[j] = bayesian_net.joint_prob(X_test[j])
            likelihoods[i] = np.sum(np.log(probs)) - lambd * np.sum(dag)**regularizer_order
        return likelihoods
        '''

    return out
