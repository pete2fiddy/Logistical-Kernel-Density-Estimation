import numpy as np
from .BayesianNetwork import BayesianNetwork

class KDEBayesianNetwork(BayesianNetwork):

    def __init__(self, dag, training_data):
        BayesianNetwork.__init__(self, dag)
