from lyssa.utils import fast_dot, gen_batches
import numpy as np
import sys
from .utils import init_dictionary, norm_cols, approx_error
from itertools import cycle
from lyssa.utils import set_openblas_threads, get_mmap

"""
This module implemented the Online Dictionary Learning
algorithm of
"Online Dictionary Learning for Sparse Coding" of Mairal et al.

It solves argmin_{D,Z} ||X-DZ||_{F}^{2}
with a sparsity constraint, using a coordinate descent procedure.
"""


def online_dict_learn(X, n_atoms, sparse_coder=None, batch_size=None, A=None, B=None, D_init=None,
                      beta=None, n_epochs=1, verbose=False, n_jobs=1, non_neg=False, mmap=False):
    """
    X: the data matrix of shape (n_features,n_samples)
    n_atoms: the number of atoms in the dictionary
    sparse_coder: must be an instance of the sparse_coding.sparse_encoder class
    batch_size: the number of datapoints in each iteration
    D_init: the initial dictionary. If None, we initialize it with randomly
            selected datapoints.
    eta: the learning rate
    mu:  the mutual coherence penalty
    n_epochs: the number of times we iterate over the dataset
    non_neg: if set to True, it uses non-negativity constraints
    n_jobs: the number of CPU threads
    mmap: if set to True, the algorithm applies memory mapping to save memory

    Note that a	large batch_size implies
    faster execution but high memory overhead, while
    a smaller batch_size implies
    slower execution but low memory overhead
    """

    # dont monitor sparse coding
    sparse_coder.verbose = False
    n_features, n_samples = X.shape
    # initialize using the data
    if D_init is None:
        D, unused_data = init_dictionary(X, n_atoms, method='data', return_unused_data=True)
    else:
        D = D_init
    print "dictionary initialized"
    if mmap:
        D = get_mmap(D)

    batch_idx = gen_batches(n_samples, batch_size=batch_size)
    n_batches = len(batch_idx)
    n_iter = n_batches
    n_total_iter = n_epochs * n_iter
    _eps = np.finfo(float).eps

    if n_jobs > 1:
        set_openblas_threads(n_jobs)

    if A is None and B is None:
        A = np.zeros((n_atoms, n_atoms))
        B = np.zeros((n_features, n_atoms))

    if beta is None:
        # create a sequence that converges to one
        beta = np.linspace(0, 1, num=n_iter)
    else:
        beta = np.zeros(n_iter) + beta

    max_patience = 10
    error_curr = 0
    error_prev = 0
    patience = 0
    approx_errors = []
    incs = []
    for e in range(n_epochs):
        # cycle over the batches
        for i, batch in zip(range(n_iter), cycle(batch_idx)):
            X_batch = X[:, batch]
            # sparse coding step
            Z_batch = sparse_coder(X_batch, D)
            # update A and B
            A = beta[i] * A + fast_dot(Z_batch, Z_batch.T)
            B = beta[i] * B + fast_dot(X_batch, Z_batch.T)
            if verbose:
                progress = float((e * n_iter) + i) / n_total_iter
                sys.stdout.write("\r" + "dictionary learning" + "...:%3.2f%%" % (progress * 100))
                sys.stdout.flush()

            DA = fast_dot(D, A)
            # this part could also be parallelized w.r.t the atoms
            for k in xrange(n_atoms):
                D[:, k] = (1 / (A[k, k] + _eps)) * (B[:, k] - DA[:, k]) + D[:, k]
            # enforce non-negativity constraints
            if non_neg:
                D[D < 0] = 0
            D = norm_cols(D)
        # replace_unused_atoms(A,unused_data,i)

        if e < n_epochs - 1:
            if patience >= max_patience:
                return D, A, B
            print ""
            print "end of epoch {0}".format(e)
            error_curr = 0
            for i, batch in zip(range(n_iter), cycle(batch_idx)):
                X_batch = X[:, batch]
                # sparse coding step
                Z_batch = sparse_coder(X_batch, D)
                error_curr += approx_error(D, Z_batch, X_batch, n_jobs=n_jobs)
            if verbose:
                print ""
                print "error:", error_curr
                print "error difference:", (error_curr - error_prev)
                error_prev = error_curr
            if (e > 0) and (error_curr > 0.9 * error_prev or error_curr > error_prev):
                patience += 1

    if verbose:
        sys.stdout.write("\r" + "dictionary learning" + "...:%3.2f%%" % (100))
        sys.stdout.flush()
        print ""
    return D, A, B


class online_dictionary_coder():
    """
    a wrapper of the online_dict_learn function
    """

    def __init__(self, n_atoms=None, sparse_coder=None, batch_size=None, beta=None, D_init=None,
                 n_epochs=1, verbose=False, memory="low", mmap=False, non_neg=False, n_jobs=1):
        self.n_atoms = n_atoms
        self.sparse_coder = sparse_coder
        self.batch_size = batch_size
        self.beta = beta
        self.n_epochs = n_epochs
        self.A = None
        self.B = None
        self.D_init = D_init
        self.memory = memory
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.mmap = mmap
        self.non_neg = non_neg

    def __call__(self, X):
        self.fit(X)
        return self.encode(X)

    def fit(self, X):
        self.D, self.A, self.B = online_dict_learn(X, self.n_atoms, sparse_coder=self.sparse_coder,
                                                   batch_size=self.batch_size, A=self.A, B=self.B, D_init=self.D_init,
                                                   beta=self.beta, n_epochs=self.n_epochs, verbose=self.verbose,
                                                   n_jobs=self.n_jobs, non_neg=self.non_neg, mmap=self.mmap)

    def encode(self, X):
        Z = self.sparse_coder(X, self.D)
        return Z
