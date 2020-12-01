'''
TODO: sampling from a general bayesian network is done in a very similar fashion --
that is, first values for nodes with no in-edges must be sampled, then the nodes
to which they point can be sampled, etc. In essence, a node cannot be sampled from
unless all its parents have been sampled already, then p(x[i] | pa(x[i])) can be
sampled from as all conditional factors are known.

Once bayesian network framework is implemented, define a general-purpose sampling
algorithm on bayesian networks,then instead convert this to a function that randomly
generates a linear-gaussian bayesian network from which samples are easily taken
'''
def generate_sparse_linear_gaussian_dataset(dims, max_deg, variance_bounds):
    U = np.zeros(dims, dims)
    
