#!/usr/bin/env python
# coding: utf-8

# In[149]:


import numpy as np
from abc import ABC, abstractmethod

# In[57]:


"""
abstract representation of a bayesian network
"""
class BayesianNet(ABC):
    __dag = np.array(np.empty)
    __roots = None
    __leaves = None
    __d = None

    def __construct_roots(self):
        """
        returns: roots of the baysian network
        """
        roots = set()
        for i in range(self.__d):
            if np.prod(self.__dag[:,i] == 0) == 1 and np.prod(self.__dag[i,:] == 0) == 0:
                roots.add(i)
        return np.array([i for i in roots]).astype(np.int64)

    def __construct_leaves(self):
        """
        returns: leaves of the bayesian network
        """
        leaves = set()
        for i in range(self.__d):
            if np.prod(self.__dag[:,i] == 0) == 0 and np.prod(self.__dag[i,:] == 0) == 1:
                leaves.add(i)
        return np.array([i for i in leaves]).astype(np.int64)

    def get_parents(self, i):
        """
        returns: the parents of the node as a set. If a node has no parents returns empty set
        """
        '''
        parents = set()
        for j in range(self.__d):
            if (self.__dag[j,i]) == 1:
                parents.add(j)
        return parents
        '''
        return np.where(self.__dag[:,i] == 1)[0]

    def get_children(self, i):
        return np.where(self.__dag[i] == 1)[0]

    def get_roots(self):
        return self.roots.copy()

    def get_d(self):
        return self.__d

    def __init__(self, __dag):
        """
        Takes in a Directed Acyclic Graph of the form
        np.array([[...]])
        a square matrix ie __dag.shape[0] = __dag.shape[1]
        where value i,j represents whether there is a directed edge from node i to node j
        Precondition: __dag is a valid Directed Acyclic Graph
        """
        self.__dag = __dag
        self.__d = self.__dag.shape[1]
        self.__roots = self.__construct_roots()
        self.__leaves = self.__construct_leaves()



    @abstractmethod
    def conditional_prob(self, i, x_i_value, parent_values):
        """
        returns: the conditional probability P(x_i = x_i_value | x_i_parents = parent_values),
        where parent_values is a dictionary whose keys are the parent indices and its values
        are their corresponding values.
        """
        pass

    def joint_prob(self,x):
        """
        returns: the joint prob of x
        """
        assert x.shape[0] == self.__d
        acc = 1
        for i in range(self.__d):
            i_parents = self.get_parents(i)
            i_parent_values = x[i_parents]
            parent_value_dict = {i_parents[i]:i_parent_values[i] for i in range(len(i_parents))}
            acc = acc * self.conditional_prob(i, x[i], parent_value_dict)
        return acc

    '''
    Not critical for implementation, if not implemented, have it raise a NotImplementedError.
    Figured it made sense to have it implemented within the bayesian net itself because some
    nets can leverage very specific and fast sampling algorithms that must be more specifically-
    defined than general-purpose sampling algorithms would allow.

    Returns a random sample for x[i] given the value of its conditioned-upon RV's in parent_value_dict.
    Assumes parent_value_dict contains values for all parents of x[i] in the network.
    '''
    @abstractmethod
    def sample_variable(self, i, parent_values):
        pass



    '''
    returns n sample from the bayesian network, net. sampler_func is a function
    that can sample from an arbitrary one-dimensional probability distribution, as a list of
    lists whose rows are the samples.

    '''
    def get_joint_samples(self, n):
        out = [None for i in range(n)]
        for i in range(len(out)):
            out[i] = self.__get_joint_sample()
        return out



    '''
    returns a sample from the bayesian network, net. sampler_func is a function
    such that sampler_func(net, i, parent_value_dict) returns a sample for x[i]
    given the conditioned-upon values parent_value_dict.
    '''
    def __get_joint_sample(self):
        x = [None for i in range(self.__d)]
        unfilled_values = np.ones(self.__d, dtype = np.bool)
        parents = [self.get_parents(i) for i in range(self.__d)]

        #samples values such that, at any point, all the conditioned-upon values are known
        #at the time of sampling. This is possible because bayesian nets are DAGs

        #this is O(d^2) treating the sampling function as constant time
        #if one were smart about the order in which values of x are populated, there should be
        #an O(d) implementation if speed becomes an issue
        while unfilled_values.any():
            for i in range(self.__d):
                if unfilled_values[i] and (~unfilled_values[parents[i]]).all():
                        parent_value_dict = {j:x[j] for j in parents[i]}
                        x[i] = self.sample_variable(i, parent_value_dict)
                        unfilled_values[i] = False
                        break
        return x
