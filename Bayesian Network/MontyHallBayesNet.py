#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import BayesianNetwork


# In[4]:


class MontyBayesNet(BayesianNetwork.BayesianNet):
    """
    Bayesian network for the monty hall problem
    """
    def conditional_prob(self,x,i):
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
            guest = x[0]
            prize = x[1]
            monty = x[2]
            if monty == guest or monty == prize:
                return 0
            else:
                if guest == prize:
                    return 1/2
                else:
                    return 1
                
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
            
"""
DAG for monty hall problem
0: guestdoor
1: prizedoor
2: montydoor
"""
monty_dag = np.array([[0,0,1],
                     [0,0,1],
                     [0,0,0]])


# In[5]:


x = MontyBayesNet(monty_dag)
#Guest picks A, Prize was under B, Monty shows A, Not possible
print(x.joint_prob(np.array(['A','B','A'])))
#Guest picks A, Prize was under A, Monty will show either B or C
print(x.joint_prob(np.array(['A','A','B'])))
#Guest picks A, Prize was under B, Monty will always show C
print(x.joint_prob(np.array(['A','B','C'])))
x.predict_prob(np.array(['A',None,'B']),1,['A','B','C'])

