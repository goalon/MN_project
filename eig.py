"""Mateusz Bajorek, 385122"""


from typing import Tuple, List
from numpy import array, outer, identity, loadtxt
from numpy.random import rand
from numpy.linalg import norm, solve


eps = 10**(-8)


def check_condition(A: array, power_matrix: array, eigvec: array, eigval: float) -> array:
    return power_matrix @ eigvec - A @ (eigval * eigvec)


def transform_eigenvector(eigvec: array, eigvecs: array, eigvals: array) -> array:
    eigval = eigvals[-1]

    for prev_eigvec, prev_eigval in zip(reversed(eigvecs[:-1]), reversed(eigvals[:-1])):
        eigvec = (eigval - prev_eigval) * eigvec + prev_eigval * (prev_eigvec.T @ eigvec) * prev_eigvec
        eigvec = eigvec / norm(eigvec)

    return eigvec


def eigenpair(A: array, power_matrix: array, inv: bool = False) -> Tuple[array, float]:
    eigvec = rand(A.shape[1])

    eigvec = solve(A, power_matrix @ eigvec) if inv else power_matrix @ eigvec
    eigval = norm(eigvec)
    eigvec = eigvec / eigval
    sign = 1

    while norm(check_condition(A, power_matrix, eigvec, sign * eigval)) > eps:
        eigvec = solve(A, power_matrix @ eigvec) if inv else power_matrix @ eigvec
        eigval = norm(eigvec)
        eigvec = eigvec / eigval
        sign *= -1

    if norm(check_condition(A, power_matrix, eigvec, -eigval)) <= eps:
        eigval = -eigval

    return eigvec, eigval


def compute_eigenpairs(A: array, n: int, inv: bool = False) -> Tuple[List[array], List[float]]:
    id_matrix = identity(A.shape[1])
    S = id_matrix * 0
    eigvec, eigval = eigenpair(A if inv else id_matrix, id_matrix - A @ S if inv else A - S, inv)
    eigvecs = [eigvec]
    eigvals = [eigval]
    eigvecs_mod = [eigvec]

    for i in range(n - 1):
        S = S + outer(eigval * eigvec, eigvec.T)
        eigvec, eigval = eigenpair(A if inv else id_matrix, id_matrix - A @ S if inv else A - S, inv)
        eigvals.append(eigval)
        eigvecs_mod.append(eigvec)
        eigvecs.append(transform_eigenvector(eigvec, eigvecs_mod, eigvals))

    return eigvecs, eigvals


A = loadtxt("test.csv")
print(A)
print(compute_eigenpairs(A, 5))
