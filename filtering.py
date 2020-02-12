import networkx as nx
import numpy as np
import scipy.sparse as sp

from utilities import sparse_2norm


def low_pass_poly_coeffs(order=3):
    spectrum = np.arange(-1, 1, 0.01)
    coefs = np.polyfit(spectrum, np.exp(-1 * spectrum), order)
    return coefs[::-1]


def sparse_matrix_power(L, k):
    if k == 0:
        return sp.csr_matrix(np.eye(L.shape[0]))
    elif k == 1:
        return L
    else:
        return L @ sparse_matrix_power(L, k - 1)


class PolynomialGraphFilter:
    def __init__(self, coefs):
        self.coefs = coefs

    def __call__(self, L):
        L = L - sp.identity(L.shape[0])
        sol = 0
        for k, coef in enumerate(self.coefs):
            sol += coef * sparse_matrix_power(L, k)
        return sol


def filter_distance(G, Gp, g):
    L = nx.normalized_laplacian_matrix(G)
    Lp = nx.normalized_laplacian_matrix(Gp)
    return sparse_2norm(g(L) - g(Lp))
