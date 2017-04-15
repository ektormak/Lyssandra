from __future__ import division
import numpy as np
from lyssa.utils import fast_dot
from functools import partial
from lyssa.utils import get_mmap, run_parallel

"""
implementations of feature encoders from the paper
"The Importance of Encoding Versus Training with Sparse Coding
and Vector Quantization"
"""


def sign_splitting(X, D, sparse_coder=None):
    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = sparse_coder.encode(X, D)
    Z_new = np.zeros((n_atoms * 2, n_samples))
    pos_idx = np.where(Z > 0)[0]
    neg_idx = np.where(Z < 0)[0]
    Z_new[pos_idx[0], pos_idx[1]] = Z[pos_idx[0], pos_idx[1]]
    Z_new[neg_idx[0], neg_idx[1]] = -Z[neg_idx[0], neg_idx[1]]
    return Z_new


def soft_thresholding(Alpha, nonzero_percentage=None, n_nonzero_coefs=None):
    n_atoms, n_samples = Alpha.shape
    # A = fast_dot(D.T,X)
    Z = np.zeros((n_atoms, n_samples))
    if nonzero_percentage is not None:
        n_nonzero_coefs = int(np.floor(nonzero_percentage * n_atoms))

    for i in xrange(n_samples):
        idx = Alpha[:, i].argsort()[::-1][:n_nonzero_coefs]
        Z[idx, i] = Alpha[idx, i]

    return Z


class feature_encoder(object):
    def __init__(self, algorithm=None, params=None, n_jobs=1, verbose=True, mmap=False):

        self.algorithm = algorithm
        self.params = params
        if self.params is None:
            self.params = {}
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.mmap = mmap

    def encode(self, X, D):
        return self.__call__(X, D)

    def __call__(self, X, D):
        from lyssa.utils import set_openblas_threads
        if self.verbose:
            msg = "feature encoding"
        else:
            msg = None

        n_atoms = D.shape[1]
        n_samples = X.shape[1]

        n_batches = 100
        if self.n_jobs > 1:
            set_openblas_threads(self.n_jobs)

        if self.algorithm == 'soft_thresholding':
            Alpha = fast_dot(D.T, X)
            data = Alpha
            args = None
            batched_args = None
            func = partial(soft_thresholding, nonzero_percentage=self.params.get('nonzero_percentage'),
                           n_nonzero_coefs=self.params.get('n_nonzero_coefs'))

        if self.n_jobs > 1:
            # disable openblas to
            # avoid hanging problem
            set_openblas_threads(1)

        Z = run_parallel(func=func, data=data, args=args, batched_args=batched_args,
                         result_shape=(n_atoms, n_samples), n_batches=n_batches,
                         mmap=self.mmap, msg=msg, n_jobs=self.n_jobs)

        # restore the previous setting
        if self.n_jobs > 1:
            set_openblas_threads(self.n_jobs)

        return Z
