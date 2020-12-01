import numpy as np

'''
A: is an n x n matrix.
Outputs: the nearest permutation matrix, P, to A found via least squares
'''
'''
A DP algorithm which examines the subproblem of determining the nearest
k x k permutation matrix to A[:k, :k]. Were the algorithm recursive,
here is what the spec for the recursive sub-problem-solver function would look like:


Let P[i] = e_{x[i]}. Then, returns x_1,...,x_k satisfying:
argmin_{k x k permutation matrix P} ||A[0:k, 0:k] - P||_F^2

As P is a permutation matrix, x must satisfy that there doesn't exist an i != j
in [0,...,k] such that x[i] = x[j], as this would yield a duplicated row within P.

Arguments:
- A: the matrix to be approximated by a permutation matrix. Is assumed to be square.
- q_prev: the quality of the solution for the subproblem when k = k - 1
- x_prev: the vectorized form of the solution for the subproblem when k = k - 1
- k: as described
'''
def find_nearest_permutation_matrix(A):
    assert(A.shape[0] == A.shape[1])
    x = np.zeros(A.shape[0], dtype = np.int)
    q = A[0,0]


    for k in range(1, A.shape[0]):


        best_q = None
        ind_to_set_to_k = None
        for j in range(0, k + 1):
            #an efficient way to compute the quality of leaving x_1,...,x_{k-1} as is, then
            #setting x[j] = k, x[k] = x[j]
            #MAKE SURE THIS IS CORRECT
            new_q = q - A[j, x[j]] + A[j, k] + A[k,x[j]]
            if ind_to_set_to_k is None or new_q > best_q:
                best_q = new_q
                ind_to_set_to_k = j

        x[k] = x[ind_to_set_to_k]
        x[ind_to_set_to_k] = k
        q = best_q

        print("q: ", q)



    P = np.zeros(A.shape, dtype = np.int)
    for i in range(0, P.shape[0]):
        P[i, x[i]] = 1
    assert(__is_permutation_matrix(P))
    return P

#taken from here: https://stackoverflow.com/questions/28895894/how-to-quickly-determine-if-a-matrix-is-a-permutation-matrix
def __is_permutation_matrix(P):
    P = np.asanyarray(P)
    return (P.ndim == 2 and P.shape[0] == P.shape[1] and \
            (P.sum(axis = 0) == 1).all() and \
            (P.sum(axis = 1) == 1).all() and \
            ((P == 1) | (P == 0)).all())


if __name__ == "__main__":
    A = 4 * np.random.rand(500,500)
    P = find_nearest_permutation_matrix(A)
    print("A: ", A)
    print("P: ", P)
