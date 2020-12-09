import numpy as np
from .BayesianNetwork import BayesianNetwork
from KDE.KDE import KDE

class KDEBayesianNetwork(BayesianNetwork):

    def __init__(self, dag, training_data, kernel):
        BayesianNetwork.__init__(self, dag)
        self.__init_numerator_and_denominator_kdes(training_data, kernel)


    def __init_numerator_and_denominator_kdes(self, X, kernel):
        self.__numerator_kdes = [None for i in range self.get_d()]
        self.__denominator_kdes = [None for i in range self.get_d()]
        for i in range(self.get_d()):
            pa_i = self.get_parents(i)
            X_trunc = np.zeros((X.shape[0], pa_i.shape[0] + 1))
            X_trunc[:,0] = X[:,i]
            X_trunc[:,1:] = X[:, pa_i]
            self.__numerator_kdes[i] = KDE(X_trunc, kernel)
            self.__denominator_kdes[i] = KDE(X_trunc[:,1:], kernel)


    def sample_variable(self, i, parent_values):
        raise NotImplementedError

    def conditional_prob(self, i, x_i_value, parent_values):
        x_numerator = np.empty(len(parent_values) + 1).astype(np.float64)
        x_numerator[0] = x_i_value
        pa_i = self.get_parents(i)
        for j in range(len(pa_i)):
            x_numerator[j+1] = parent_values[pa_i[j]]

        return self.__numerator_kdes[i].joint_prob(x_numerator) / self.__denominator_kdes[i].joint_prob(x_numerator[1:])
