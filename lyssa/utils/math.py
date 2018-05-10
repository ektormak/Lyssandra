from __future__ import division
import numpy as np
import scipy.linalg.blas
from scipy.linalg import get_blas_funcs

ddot = scipy.linalg.blas.ddot
dgemv = scipy.linalg.blas.dgemv
dgemm = scipy.linalg.blas.dgemm


def fast_dot(a,b):
    """a fast implementation of matrix-matrix,
       matrix-vector,vector-vector products
    """
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 2 and b_dim == 2:
        return np.dot(a,b)
    elif a_dim == 2 and b_dim == 1 or a_dim == 1 and b_dim == 2:
        # matrix to vector product
        # GEMV is slower than np.dot-why?
        return np.dot(a,b)
    elif a_dim == 1 and b_dim == 1:
        return _dot(a,b)


def outer(a,b):
    return np.outer(a, b)


def _dot(a,b):
    # the gemv BLAS function for vector-vector product
    # it is faster than np.dot
    return ddot(x=a, y=b)


def gemv(A, x):
    return dgemv(alpha=1.0, a=A, x=x)


def _force_forder(x):
    """
    Converts arrays x to fortran order. Returns
    a tuple in the form (x, is_transposed).
    """
    if x.flags.c_contiguous:
        return (x.T, True)
    else:
        return (x, False)


def norm(x):
    nrm2 = get_blas_funcs(['nrm2'], [x])[0]
    return nrm2(x)


def frobenius_squared(A):
    return np.sum(np.power(A, 2))


def normalize(x, eps=np.finfo(float).eps):
    return x / (norm(x) + eps)


def norm_cols(X, eps=np.finfo(float).eps):
    """
    normalize the columns of a matrix
    """
    norms = np.sqrt(np.einsum('ij,ij->j', X, X)) + eps
    X /= norms[np.newaxis, :]
    return X
