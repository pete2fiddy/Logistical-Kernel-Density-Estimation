import numpy as np
from bayesian_net.BayesianNetwork import BayesianNet
from KDE.KDE import KDE
from sklearn.neighbors import KernelDensity

class KDEBayesianNetwork(BayesianNet):

    def __init__(self, dag, training_data, kernel):
        BayesianNet.__init__(self, dag)
        self.__init_numerator_and_denominator_kdes(training_data, kernel)


    def __init_numerator_and_denominator_kdes(self, X, kernel):
        self.__numerator_kdes = [None for i in range(self.get_d())]
        self.__denominator_kdes = [None for i in range(self.get_d())]
        for i in range(self.get_d()):
            pa_i = self.get_parents(i)
            X_trunc = np.zeros((X.shape[0], pa_i.shape[0] + 1))
            X_trunc[:,0] = X[:,i]
            X_trunc[:,1:] = X[:, pa_i]
            self.__numerator_kdes[i] = KernelDensity(kernel = kernel).fit(X_trunc)#KDE(X_trunc, kernel)
            if X_trunc.shape[1] > 1:
                #None if RV i is not conditionally dependent on anything. In this case,
                #denominator KDE should just act as the 1 function
                self.__denominator_kdes[i] = KernelDensity(kernel = kernel).fit(X_trunc[:,1:])#KDE(X_trunc[:,1:], kernel)


    def sample_variable(self, i, parent_values):
        raise NotImplementedError

    '''
    def conditional_prob(self, i, x_i_value, parent_values):
        x_numerator = np.empty(len(parent_values) + 1).astype(np.float64)
        x_numerator[0] = x_i_value
        pa_i = self.get_parents(i)
        for j in range(len(pa_i)):
            x_numerator[j+1] = parent_values[pa_i[j]]

        log_out = self.__numerator_kdes[i].score_samples(x_numerator.reshape(1,-1))[0]#self.__numerator_kdes[i].joint_prob(x_numerator)
        if self.__denominator_kdes[i] is not None:
            log_out -= self.__denominator_kdes[i].score_samples(x_numerator[1:].reshape(1,-1))[0]#self.__denominator_kdes[i].joint_prob(x_numerator[1:])
        return np.exp(log_out)
    '''

    def conditional_prob(self, i, x_i_values, parent_values):
        
        X_numerator = np.empty((x_i_values.shape[0], len(parent_values) + 1)).astype(np.float64)
        X_numerator[:,0] = x_i_values

        pa_i = self.get_parents(i)

        for j in range(len(pa_i)):
            X_numerator[:,j+1] = parent_values[pa_i[j]]
        log_out = self.__numerator_kdes[i].score_samples(X_numerator)
        if self.__denominator_kdes[i] is not None:
            log_out -= self.__denominator_kdes[i].score_samples(X_numerator[:,1:])
        return np.exp(log_out)
