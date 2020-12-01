import numpy as np

'''
generates a random n x n permutation matrix (dtype is np.int)
'''
def random_permutation_matrix(n):
    out = np.eye(n)
    np.random.shuffle(out)
    return out.astype(np.int)
