from typing import Tuple, List
from numpy import array, outer, identity
from numpy.random import rand
from numpy.linalg import norm, solve


eps = 10**(-6)


def check_condition(A: array, power_matrix: array, eigvec: array, eigval: float) -> array:
    return power_matrix @ eigvec - A @ (eigval * eigvec)


def eigenpair(A: array, power_matrix: array) -> Tuple[array, float]:
    eigvec = rand(A.shape[1])

    eigvec = solve(A, power_matrix @ eigvec)
    eigval = norm(eigvec)
    eigvec = eigvec / eigval
    sign = 1

    while norm(check_condition(A, power_matrix, eigvec, sign * eigval)) > eps:
        eigvec = solve(A, power_matrix @ eigvec)
        eigval = norm(eigvec)
        eigvec = eigvec / eigval
        sign *= -1

    if norm(check_condition(A, power_matrix, eigvec, -eigval)) <= eps:
        eigval = -eigval

    return eigvec, eigval


def transforn_eigenvector(new_eigvec: array, new_eigval: float, old_eigvec: array, old_eigval: float) -> array:
    return (new_eigval - old_eigval) * new_eigvec + old_eigval * (old_eigvec.T @ new_eigvec) * old_eigvec


def compute_eigenpairs(A: array, n: int) -> Tuple[List[array], List[float]]:
    id_matrix = identity(A.shape[1])
    eigvec, eigval = eigenpair(A, id_matrix)
    eigvecs = [eigvec]
    eigvals = [eigval]
    S = id_matrix * 0

    for i in range(n - 1):
        S = S + outer(eigval * eigvec, eigvec.T)
        eigvec, eigval = eigenpair(A, id_matrix - A @ S)
        print(eigvec)
        print(transforn_eigenvector(eigvec, eigval, eigvecs[0], eigvals[0]))
        eigvecs.append(transforn_eigenvector(eigvec, eigval, eigvecs[0], eigvals[0]))
        eigvals.append(eigval)

    return eigvecs, eigvals


A = array([[9, 8, 3], [4, 9, 9], [2, 9, 2]])
print(compute_eigenpairs(A, 3))
