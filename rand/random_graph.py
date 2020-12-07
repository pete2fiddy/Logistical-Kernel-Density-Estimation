import numpy as np
import rand.random_matrix as random_matrix

'''
Generates an adjacency matrix for a Directed Acyclic Graph over n nodes, where
each has out-degree of at most max_out_deg.

A minimum degree argument is not included, as all DAGs must have at least one
node with degree zero (such a node must exist to be placed at the end of the
topological sort of the DAG)
'''
def random_dag(n, max_out_deg):
    U = np.zeros((n, n), dtype = np.int)
    for i in range(0, U.shape[0] - 1):
        i_out_deg = np.random.randint(0, min(U.shape[1] - i, max_out_deg))
        i_children = np.random.choice(np.arange(i + 1, U.shape[0]), size = i_out_deg, replace = False)
        U[i, i_children] = 1
    P = random_matrix.random_permutation_matrix(n)
    return np.dot(np.dot(P.T, U), P)
