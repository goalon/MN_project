from typing import Tuple
from numpy import array, identity
from numpy.random import rand
from numpy.linalg import norm


eps = 10**(-6)


def eigenpair(A: array) -> Tuple[array, float]:
    eigvec = rand(A.shape[1])
    eigvec_next = A @ eigvec
    eigval = eigvec_next.T @ eigvec_next

    while norm(eigvec_next - eigval * eigvec) > eps:
        eigvec = eigvec_next / norm(eigvec_next)
        eigvec_next = A @ eigvec
        eigval = eigvec_next.T @ eigvec

    return eigvec, eigval


A = array([[1, 2], [3, 4]])
eig_v, eig_n = eigenpair(A)
B = A - eig_n * identity(A.shape[1])
eig_v_2, eig_n_2 = eigenpair(B)
print(eig_v_2, eig_n_2 + eig_n)
