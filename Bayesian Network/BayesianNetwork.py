#!/usr/bin/env python
# coding: utf-8

# In[149]:


import numpy as np


# In[57]:


"""
abstract representation of a bayesian network
"""
class BayesianNet:
    dag = np.array(np.empty)
    roots = None
    leaves = None
    d = None
    nodes_in = None
    
    def get_roots(self):
        """
        returns: roots of the baysian network
        """
        roots = set()
        for i in range(self.d):
            if np.prod(self.dag[:,i] == 0) == 1 and np.prod(self.dag[i,:] == 0) == 0:
                roots.add(i)
        return roots
    
    def get_leaves(self):
        """
        returns: leaves of the bayesian network
        """
        leaves = set()
        for i in range(self.d):
            if np.prod(self.dag[:,i] == 0) == 0 and np.prod(self.dag[i,:] == 0) == 1:
                leaves.add(i)
        return leaves
    
    def get_parents(self, i):
        """
        returns: the parents of the node as a set. If a node has no parents returns empty set
        """
        parents = set()
        for j in range(self.d):
            if (self.dag[j,i]) == 1:
                parents.add(j)
        return parents
    
    def nodes_in_graph(self):
        """
        returns: the set of nodes in the graph
        """
        nodes_in = set(range(self.d))
        for i in range(self.d):
            if np.prod(self.dag[:,i] == 0) == 1 and np.prod(self.dag[i,:] == 0) == 1:
                nodes_in.remove(i)
        return nodes_in
    
    def in_graph(self, i):
        """
        returns: true if the node is in the graph
        """
        return i in self.nodes_in
    
    def __init__(self, dag):
        """
        Takes in a Directed Acyclic Graph of the form 
        np.array([[...]])
        a square matrix ie dag.shape[0] = dag.shape[1]
        where value i,j represents whether there is a directed edge from node i to node j
        Precondition: dag is a valid Directed Acyclic Graph
        """
        self.dag = dag
        self.d = self.dag.shape[1]
        self.roots = self.get_roots()
        self.leaves = self.get_leaves()
        self.nodes_in = self.nodes_in_graph()
        
    def conditional_prob(self,x,i):
        """
        conditional probavility. use get_parents() if needed
        """
        pass  
    
    def joint_prob(self,x):
        """
        returns: the joint prob of x
        """
        assert x.shape[0] == self.d
        acc = 1
        for i in range(self.d):
            acc = acc * self.conditional_prob(x,i)
        return acc
    
    def predict_prob(self,x,i,vals):
        """
        returns: a prediction of the unspecifiec value x[i]
        vals is a list of possible values that x[i] can take
        """
        pass

