import numpy as np
import simulated_annealing as sa

'''
is j an ancestor of i in adjacency matrix dag A
'''
def isAncestor(A, j, i):
    checked = np.zeros(A.shape[0], dtype=bool)
    stack = [i]
    while(stack):
        current = stack.pop()
        if current == j:
            return True
        for parent in range(0, A.shape[0]):
            if A[parent, current]==1 and not checked[parent]:
                stack.append(parent)
                checked[parent]=True
    return False


'''
Get edge additions and deletion possibilities based on input matrix A
where A is an adjacency 2d numpy matrix of a DAG
'''
def neighbors_func(A):
    assert(A.shape[0] == A.shape[1])
    possibilities_list = []
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[0]):
            if A[i, j]!=0:
                #Produce neighbors with deleted edges
                B= np.empty_like(A)
                B[:] = A
                B[i, j]=0
                possibilities_list.append(B)
                print(B)
            else:
                #Checks and produces neighbors with added edges
                if not isAncestor(A, j, i):
                    B= np.empty_like(A)
                    B[:] = A
                    B[i, j]=1
                    possibilities_list.append(B)
                    
def exponentiation(A, exponent):
    if(exponent == 1):
        return A
    else:
        final_matrix = exponentiation(A, (exponent-(exponent % 2))/2)
        final_matrix = np.matmul(final_matrix, final_matrix)
        if(exponent % 2 == 1):
            final_matrix = np.matmul(final_matrix, A)
    return final_matrix

def check_dag_properties(A):
    assert(A.shape[0] == A.shape[1])
    exponent = A.shape[0]
    final_matrix = exponentiation(A, exponent)
    if np.trace(final_matrix) == 0:
        return True
    else:
        return False
if __name__ == "__main__":
    a = np.array([[0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 1, 0]])
    assert(check_dag_properties(a))
    print(neighbors_func(a))
