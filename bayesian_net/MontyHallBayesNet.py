#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from bayesian_net.BayesianNetwork import BayesianNet


# In[4]:

'''
TODO: update this to new spec

DEPRECATED, written before bayesian nets were vectorized
'''
class MontyBayesNet(BayesianNet):


    """
    DAG for monty hall problem
    0: guestdoor
    1: prizedoor
    2: montydoor
    """

    def __init__(self):
        BayesianNet.__init__(self, np.array([[0,0,1],
                             [0,0,1],
                             [0,0,0]]))



    def conditional_prob(self, i, x_i_value, parent_values):
        """
        assuming the door chosen by guest and door with prize are chosen uniformly at random
        Monty will always pick a door that the guest did not to reveal, and will never reveal the prize door
        doors are labeled 'A', 'B', 'C'
        """
        if i == 0:
            return 1/3
        elif i == 1:
            return 1/3
        else:
            guest = parent_values[0]
            prize = parent_values[1]
            monty = x_i_value

            if monty == guest or monty == prize:
                return 0
            else:
                if guest == prize:
                    return 1/2
                else:
                    return 1



    """
    returns: a prediction of the unspecifiec value x[i]
    vals is a list of possible values that x[i] can take
    """
    def predict_prob(self,x,i,vals):
        probs = {}
        total_prob = 0
        for val in vals:
            x[i] = val
            prob = self.joint_prob(x)
            probs[val] = prob
            total_prob += prob
        probs = {k:p/total_prob for k, p in probs.items()}
        return probs

    '''
    For use with bayesian_net_sampler
    '''
    def sample_variable(self, i, parent_values):
        probs = {k: self.conditional_prob(i, k, parent_values) for k in ['A', 'B', 'C']}
        r = np.random.random_sample()
        sum = 0
        for k in probs:
            p = probs[k]
            if r >= sum and r < sum + p:
                return k
            sum += p
        #shouldn't get here
        return None
