from rand import random_graph as random_graph
import numpy as np

from bayesian_net.MontyHallBayesNet import MontyBayesNet
from rand.sampling import bayesian_net_sampler




x = MontyBayesNet()
#Guest picks A, Prize was under B, Monty shows A, Not possible
print(x.joint_prob(np.array(['A','B','A'])))
#Guest picks A, Prize was under A, Monty will show either B or C
print(x.joint_prob(np.array(['A','A','B'])))
#Guest picks A, Prize was under B, Monty will always show C
print(x.joint_prob(np.array(['A','B','C'])))
x.predict_prob(np.array(['A',None,'B']),1,['A','B','C'])

samples = bayesian_net_sampler.get_samples(5000, x, x.monty_hall_sampler_func, dtype = np.str)
print("samples: \n", samples)
