import numpy as np
from bayesian_net.BayesianNetwork import BayesianNet
from scipy import stats

'''
See the linear-gaussian section of Bishop's Pattern Recognition and Machine Learning
'''
class LinearGaussianBayesianNet(BayesianNet):

    def __init__(self, W, biases, std_devs):
        BayesianNet.__init__(self, self.__construct_dag_from_W(W))
        self.__W = W
        self.__biases = biases
        self.__std_devs = std_devs

    def __construct_dag_from_W(self, W):
        return (W.T != 0).astype(np.int)

    def __get_node_mean(self, i, parent_values):
        mu = self.__biases[i]
        for j in parent_values:
            mu += self.__W[i,j] * parent_values[j]
        return mu

    def conditional_prob(self, i, x_i_values, parent_values):
        out = np.zeros(x_i_values.shape[0], dtype = np.float64)
        for j in range(out.shape[0]):
            j_parent_values = {k:parent_values[k][j] for k in parent_values}
            out[j] = stats.norm(self.__get_node_mean(i, j_parent_values), self.__std_devs[i]).pdf(x_i_values[j])
        return out

    def sample_variable(self, i, parent_values):
        return np.random.normal(loc = self.__get_node_mean(i, parent_values), scale = self.__std_devs[i])
