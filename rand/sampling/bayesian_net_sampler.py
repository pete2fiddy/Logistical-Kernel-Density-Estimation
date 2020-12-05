import numpy as np



'''
returns n sample from the bayesian network, net. sampler_func is a function
that can sample from an arbitrary one-dimensional probability distribution, as a matrix
whose rows are the samples.

Assumes all RV's in the network are of the same type dtype
'''
def get_samples(n, net, sampler_func, dtype = np.float64):
    out = np.empty((n, net.get_d())).astype(dtype)
    for i in range(out.shape[0]):
        out[i] = __get_sample(net, sampler_func, dtype = dtype)
    return out



'''
returns a sample from the bayesian network, net. sampler_func is a function
that can sample from an arbitrary one-dimensional probability distribution.

Assumes all RV's in the network are of the same type dtype
'''
def __get_sample(net, sampler_func, dtype = np.float64):
    x = np.empty(net.get_d()).astype(dtype)
    print
    unfilled_values = np.ones(x.shape[0], dtype = np.bool)
    parents = [net.get_parents(i) for i in range(x.shape[0])]

    #samples values such that, at any point, all the conditioned-upon values are known
    #at the time of sampling. This is possible because bayesian nets are DAGs

    #this is O(d^2) treating the sampling function as constant time
    #if one were smart about the order in which values of x are populated, there should be
    #an O(d) implementation if speed becomes an issue
    while unfilled_values.any():
        for i in range(x.shape[0]):
            if unfilled_values[i] and not (unfilled_values[parents[i]]).any():

                parent_value_dict = {parents[i][j]:x[j] for j in parents[i]}
                x[i] = sampler_func(__conditional_prob_wrapper_func(net, i, parent_value_dict))
                unfilled_values[i] = False
                break
    return x


def __conditional_prob_wrapper_func(net, i, parent_value_dict):
    def out(x_i):
        return net.conditional_prob(i, x_i, parent_value_dict)
    return out
