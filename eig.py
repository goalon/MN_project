from typing import Tuple
from numpy import array, identity
from numpy.random import rand
from numpy.linalg import norm


def power_iteration(A: array, iterations: int = 1000) -> Tuple[array, float]:
    v = rand(A.shape[1])

    for _ in range(iterations):
        v_next = A @ v
        v = v_next / norm(v_next)

    return v, ((A @ v).T @ v) / (v.T @ v)


A = array([[1, 2], [3, 4]])
eig_v, eig_n = power_iteration(A)
B = A - eig_n * identity(A.shape[1])
eig_v_2, eig_n_2 = power_iteration(B)
print(eig_v_2, eig_n_2 + eig_n)
