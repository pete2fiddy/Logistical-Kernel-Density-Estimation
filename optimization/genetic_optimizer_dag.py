
import numpy as np

def dag_crossover_fast(A, B):
    n_mutual_edges = np.sum(A * B)
    n_non_mutual_edges = np.sum(A + B - 2*(A * B))
    n_edges = n_mutual_edges + int(n_non_mutual_edges / 2)

    mutual_edges = np.column_stack(np.where((A * B) >= 1))
    non_mutual_edges = np.column_stack(np.where(A + B - 2*(A * B) >= 1))

    E = [e for e in non_mutual_edges]
    for e in mutual_edges:
        E.append(e)

    Q = np.zeros(A.shape, dtype = np.int)
    N = np.zeros(A.shape, dtype = np.int)
    while n_edges > 0 and len(E) > 0:
        i = None
        if n_edges >= n_non_mutual_edges:
            i = len(E) - 1
        else:
            i = np.random.randint(len(E))
        e = E.pop(i)

        #don't need to check if adding edge is DAG-legal if n_edges >= n_non_mutual_edges,
        #as in this case, e is an edge that both A and B share
        if n_edges >= n_non_mutual_edges or __can_add_edge(Q, e):
            Q =__update_Q(Q, e)
            N[e[0], e[1]] = 1
            n_edges -= 1

    assert(__is_dag(N))
    return N


def __can_add_edge(Q, e):
    (u,v) = e
    if u == v or np.trace(Q) != 0:
        return False

    if np.dot(Q[:,u], Q[v]) != 0 or\
        Q[v,u] == 1 or\
        Q[u,v] == 1:
        return False
    return True



def __update_Q(Q, e):
    (u,v) = e
    out = Q.copy()
    out[u,v] = 1
    out += np.outer(Q[:,u], Q[v])
    out[:, v] += Q[:, u]
    out[u] += Q[v]
    out = (out >= 1).astype(np.int)
    return out

def __is_dag(N):
    #returns whether N is nilpotent. Assumes N is a 0-1 matrix
    return np.sum(np.linalg.matrix_power(N, N.shape[0] + 1)) == 0

'''
returns: a function, f, such that f(dag) performs n_mutations mutations (i.e.
edge additions / removals) to the DAG dag, while still maintaining the DAG
property. Also modifies A in-place.
'''
def mutate_dag_func(min_mutations, max_mutations):
    def out(A):
        n_mutations = np.random.randint(min_mutations, max_mutations)
        where_edges = np.where(A != 0)
        where_not_edges = np.where(A == 0)
        A_edges = [np.array([where_edges[0][i], where_edges[1][i]]) for i in range(len(where_edges[0]))]
        not_A_edges = [np.array([where_not_edges[0][i], where_not_edges[1][i]]) for i in range(len(where_not_edges[0]))]
        for mutation_num in range(0, n_mutations):
            if np.random.rand() < 0.5:
                __delete_random_edge(A, A_edges, not_A_edges)
            else:
                __add_random_edge(A, A_edges, not_A_edges)
        return A

    return out

'''
deletes a random edge from A in-place, and updates both A_edges
and not_A_edges in-place

returns False if no edges can be removed
'''
def __delete_random_edge(A, A_edges, not_A_edges):
    if len(A_edges) == 0:
        return False
    to_delete = A_edges.pop(np.random.randint(0, len(A_edges)))
    A[to_delete[0], to_delete[1]] = 0
    not_A_edges.append(to_delete)

'''
adds a random edge to A in-place, updating A_edges and not_A_edges
in-place. Returns False if no edge addition is possible.
'''
def __add_random_edge(A, A_edges, not_A_edges):
    viable_edges = [i for i in range(0, len(not_A_edges))]
    while len(viable_edges) > 0:
        i = viable_edges.pop(np.random.randint(0, len(viable_edges)))
        to_add = not_A_edges[i]
        A[to_add[0], to_add[1]] = 1
        if __is_dag(A):
            not_A_edges.pop(i)
            A_edges.append(to_add)
            return A
        A[to_add[0], to_add[1]] = 0
    return False




'''
#old code that works, but makes DAGs that are too small
def dag_crossover(A, B):
    N = np.zeros(A.shape, dtype = np.int)
    for n in range(1, N.shape[0]):

        __expand_dag(n, A, B, N)

    #print("N: \n", N)
    #print("N^n: ", np.linalg.matrix_power(N, N.shape[0]))
    assert(__is_dag(N))

    #add in all edges that A and B both share. Not entirely sure this guarantees
    #DAG constraint as I haven't proved it yet, so I use an assertion to ensure
    #the result remains a DAG
    #N += (A * B)
    #N = (N >= 1).astype(np.int)

    print("N: \n", N)
    print("N^n: \n", np.linalg.matrix_power(N, N.shape[0] + 1))

    print("A edges: ", np.sum(A))
    print("B edges: ", np.sum(B))
    print("A * B edges: ", np.sum(A * B))
    print("N edges: ", np.sum(N))
    assert(__is_dag(N))
    return N

def __is_dag(N):
    #returns whether N is nilpotent. Assumes N is a 0-1 matrix
    return np.sum(np.linalg.matrix_power(N, N.shape[0] + 1)) == 0

def __expand_dag(n, A, B, N):
    x, y = (None, None)
    #TODO: enforce fairness between elements from columns and rows of A and B being added to the DAG
    #through changing the probabilities in which each expansion method, y = 0 or Nx = 0, is
    #used. Can even use the output of both methods for reference in choosing p such that this
    #is achieved
    p = 0.5
    if np.random.rand() < p:
        x, y = __y_eq_0_expand_dag(n, A, B, N)
    else:
        x, y = __Nx_eq_0_expand_dag(n, A, B, N)
    N[n, :n-1] = y
    N[:n-1, n] = x



def __y_eq_0_expand_dag(n, A, B, N):
    #if y is 0, then x can be anything
    y = np.zeros(n - 1, dtype = np.int)
    x = ((A[:n-1, n] + B[:n-1, n]) >= 1).astype(np.int)
    return x, y



def __Nx_eq_0_expand_dag(n, A, B, N):
    #if Nx = 0, then x can be filled wherever a column of N is 0, and y can be anything so long
    #as y^T x = 0

    #constructs a maximal x from the corresponding columns of A and B such that Nx = 0
    x = np.sum(N[:n-1, :n-1], axis = 0)
    x *= A[:n-1, n] + B[:n-1, n]
    x = (x >= 1).astype(np.int)

    #constructs a maximal y from the corresponding rows of A and B
    y = (A[n,:n-1] + B[n,:n-1]) >= 1

    #zeros out the elements of x that are causing y^T x != 0
    #x *= (1 - y)
    x *= 0
    return x, y
'''
