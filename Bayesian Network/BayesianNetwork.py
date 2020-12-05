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

    def __get_roots(self):
        """
        returns: roots of the baysian network
        """
        roots = set()
        for i in range(self.__d):
            if np.prod(self.__dag[:,i] == 0) == 1 and np.prod(self.__dag[i,:] == 0) == 0:
                roots.add(i)
        return roots

    def __get_leaves(self):
        """
        returns: leaves of the bayesian network
        """
        leaves = set()
        for i in range(self.__d):
            if np.prod(self.__dag[:,i] == 0) == 0 and np.prod(self.__dag[i,:] == 0) == 1:
                leaves.add(i)
        return leaves

    def __get_parents(self, i):
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
        self.__roots = self.__get_roots()
        self.__leaves = self.__get_leaves()

    '''
    #old implementaiton, which technically allows overriding implementation to have
    access to RV values upon which it is not conditionally dependent, so one could
    code a bayesian net that doesn't obey its own structure
    @abstractmethod
    def conditional_prob(self,x,i):
        """
        conditional probability. use get_parents() if needed
        """
        pass
    '''

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
            i_parents = self.__get_parents(i)
            i_parent_values = x[i_parents]
            parent_value_dict = {i_parents[i]:i_parent_values[i] for i in range(len(i_parents))}
            acc = acc * self.conditional_prob(i, x[i], parent_value_dict)
        return acc

    def get_dag(self):
        return self.__dag.copy()
