from lyssa.utils import gen_batches
import numpy as np
import sys
from .utils import init_dictionary, norm_cols, approx_error
from itertools import cycle
from lyssa.utils import set_openblas_threads, get_mmap

"""
This module implemented the Projected Gradient Descent algorithm
for Dictionary Learning. It is described in Chapter 5 of
"Sparse Modeling for Image and Vision Processing" of Mairal et al.

It solves argmin_{D,Z} ||X-DZ||_{F}^{2} + mu*||D^{T}D-I||_{F}^{2}
with a sparsity constraint, using a gradient descent procedure.
"""


def projected_grad_desc(X, n_atoms=None, sparse_coder=None, batch_size=None, D_init=None,
                        eta=None, mu=None, n_epochs=None, non_neg=False, verbose=False, n_jobs=1, mmap=False):
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

    if eta is None:
        raise ValueError('Must specify learning rate.')

    # don't monitor sparse coding
    sparse_coder.verbose = False
    n_features, n_samples = X.shape
    # initialize the dictionary
    # with the dataset
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
    I = np.eye(n_atoms)

    if n_batches > n_iter:
        print "will iterate on only {0:.2f}% of the dataset".format((float(n_iter) / n_batches) * 100)

    if n_jobs > 1:
        set_openblas_threads(n_jobs)

    max_patience = 10
    error_prev = 0
    patience = 0
    approx_errors = []
    for e in range(n_epochs):
        # cycle over the batches
        for i, batch in zip(range(n_iter), cycle(batch_idx)):
            X_batch = X[:, batch]
            # sparse coding step
            Z_batch = sparse_coder(X_batch, D)

            if verbose:
                progress = float((e * n_iter) + i) / n_total_iter
                sys.stdout.write("\r" + "dictionary learning" + "...:%3.2f%%" % (progress * 100))
                sys.stdout.flush()

            # the gradient of the approximation error
            grad_approx = np.dot(np.dot(D, Z_batch) - X_batch, Z_batch.T)
            # the gradient of the incoherence penalty
            if mu is not None and mu > 0:
                grad_incoh = 2 * mu * np.dot(D, np.dot(D.T, D) - I)
            else:
                grad_incoh = 0

            grad = grad_approx
            D = D - (eta * grad) + grad_incoh
            # enforce non-negativity
            if non_neg:
                D[D < 0] = 0
            # project to l2 unit sphere
            D = norm_cols(D)
            # sparse coding
            Z = sparse_coder(X, D)
        #replace_unused_atoms(A,unused_data,i)

        if e < n_epochs - 1:
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
            if patience >= max_patience:
                return D
    if verbose:
        sys.stdout.write("\r" + "dictionary learning" + "...:%3.2f%%" % (100))
        sys.stdout.flush()
        print ""
    return D


class dictionary_learner():
    """
    a wrapper of the projected_grad_desc function
    """

    def __init__(self, n_atoms=None, sparse_coder=None, batch_size=None, eta=None, mu=None, D_init=None,
                 n_epochs=1, verbose=False, memory="low", mmap=False, non_neg=False, n_jobs=1):
        self.n_atoms = n_atoms
        self.sparse_coder = sparse_coder
        self.batch_size = batch_size
        self.eta = eta
        self.mu = mu
        self.n_epochs = n_epochs
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
        self.D = projected_grad_desc(X, n_atoms=self.n_atoms, sparse_coder=self.sparse_coder,
                                     batch_size=self.batch_size, mu=self.mu, D_init=self.D_init,
                                     eta=self.eta, n_epochs=self.n_epochs, verbose=self.verbose, n_jobs=self.n_jobs,
                                     non_neg=self.non_neg, mmap=self.mmap)

    def encode(self, X):
        Z = self.sparse_coder(X, self.D)
        return Z
