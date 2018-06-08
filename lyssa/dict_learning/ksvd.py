from __future__ import division
import numpy as np
import time
from lyssa.utils import get_mmap
from lyssa.utils.math import fast_dot, norm, normalize
from .utils import approx_error, force_mi, average_mutual_coherence
from sklearn.utils.extmath import randomized_svd
import warnings
import sys
from lyssa.utils import set_openblas_threads

"""
This module implements the KSVD algorithm of "K-SVD: An Algorithm for Designing Overcomplete
Dictionaries for Sparse Representation" of Aharon et al.
Users should call the  ksvd_dict_learn function or create an instance of the ksvd_coder class
"""


def ksvd(Y, D, X, n_cycles=1, verbose=True):
    n_atoms = D.shape[1]
    n_features, n_samples = Y.shape
    unused_atoms = []
    R = Y - fast_dot(D, X)

    for c in range(n_cycles):
        for k in range(n_atoms):
            if verbose:
                sys.stdout.write("\r" + "k-svd..." + ":%3.2f%%" % ((k / float(n_atoms)) * 100))
                sys.stdout.flush()
            # find all the datapoints that use the kth atom
            omega_k = X[k, :] != 0
            if not np.any(omega_k):
                unused_atoms.append(k)
                continue
            # the residual due to all the other atoms but k
            Rk = R[:, omega_k] + np.outer(D[:, k], X[k, omega_k])
            U, S, V = randomized_svd(Rk, n_components=1, n_iter=10, flip_sign=False)
            D[:, k] = U[:, 0]
            X[k, omega_k] = V[0, :] * S[0]
            # update the residual
            R[:, omega_k] = Rk - np.outer(D[:, k], X[k, omega_k])
        print ""
    return D, X, unused_atoms


def nn_ksvd(Y, D, X, n_cycles=1, verbose=True):
    # the non-negative variant
    n_atoms = D.shape[1]
    n_features, n_samples = Y.shape
    unused_atoms = []
    R = Y - fast_dot(D, X)

    for k in range(n_atoms):
        if verbose:
            sys.stdout.write("\r" + "k-svd..." + ":%3.2f%%" % ((k / float(n_atoms)) * 100))
            sys.stdout.flush()
        # find all the datapoints that use the kth atom
        omega_k = X[k, :] != 0
        if not np.any(omega_k):
            unused_atoms.append(k)
            continue
        # the residual due to all the other atoms but k
        Rk = R[:, omega_k] + np.outer(D[:, k], X[k, omega_k])
        try:
            U, S, V = randomized_svd(Rk, n_components=1, n_iter=50, flip_sign=False)
        except:
            warnings.warn('SVD error')
            continue

        d = U[:, 0]
        x = V[0, :] * S[0]
        # projection to the constraint set
        d[d < 0] = 0
        x[x < 0] = 0

        dTd = np.dot(d, d)
        xTx = np.dot(x, x)
        if dTd <= np.finfo('float').eps or xTx <= np.finfo('float').eps:
            continue

        for j in range(n_cycles):
            d = np.dot(Rk, x) / np.dot(x, x)
            d[d < 0] = 0
            x = np.dot(d.T, Rk) / np.dot(d, d)
            x[x < 0] = 0

        _norm = norm(d)
        d = d / _norm
        x = x * _norm
        D[:, k] = d
        X[k, omega_k] = x
        # update the residual
        R[:, omega_k] = Rk - np.outer(D[:, k], X[k, omega_k])
    print ""
    return D, X, unused_atoms


def approx_ksvd(Y, D, X, n_cycles=1, verbose=True):
    # the approximate KSVD algorithm
    n_atoms = D.shape[1]
    n_features, n_samples = Y.shape
    unused_atoms = []
    R = Y - fast_dot(D, X)

    for c in range(n_cycles):
        for k in range(n_atoms):
            if verbose:
                sys.stdout.write("\r" + "k-svd..." + ":%3.2f%%" % ((k / float(n_atoms)) * 100))
                sys.stdout.flush()
            # find all the datapoints that use the kth atom
            omega_k = X[k, :] != 0
            if not np.any(omega_k):
                # print "this atom is not used"
                unused_atoms.append(k)
                continue
            Rk = R[:, omega_k] + np.outer(D[:, k], X[k, omega_k])
            # update of D[:,k]
            D[:, k] = np.dot(Rk, X[k, omega_k])
            D[:, k] = normalize(D[:, k])
            # update of X[:,k]
            X[k, omega_k] = np.dot(Rk.T, D[:, k])
            # update the residual
            R[:, omega_k] = Rk - np.outer(D[:, k], X[k, omega_k])
        print ""

    return D, X, unused_atoms


def ksvd_dict_learn(X, n_atoms, init_dict='data', sparse_coder=None,
                    max_iter=20, non_neg=False, approx=False, eta=None,
                    n_cycles=1, n_jobs=1, mmap=False, verbose=True):
    """
    The K-SVD algorithm

    X: the data matrix of shape (n_features,n_samples)
    n_atoms: the number of atoms in the dictionary
    sparse_coder: must be an instance of the sparse_coding.sparse_encoder class
    approx: if true, invokes the approximate KSVD algorithm
    max_iter: the maximum number of iterations
    non_neg: if set to True, it uses non-negativity constraints
    n_cycles: the number of updates per atom (Dictionary Update Cycles)
    n_jobs: the number of CPU threads
    mmap: if set to True, the algorithm applies memory mapping to save memory
    """
    n_features, n_samples = X.shape
    shape = (n_atoms, n_samples)
    Z = np.zeros(shape)
    # dictionary initialization
    # track the datapoints that are not used as atoms
    unused_data = []
    if init_dict == 'data':
        from .utils import init_dictionary
        D, unused_data = init_dictionary(X, n_atoms, method=init_dict, return_unused_data=True)
    else:
        D = np.copy(init_dict)

    if mmap:
        D = get_mmap(D)
        sparse_coder.mmap = True

    print "dictionary initialized"
    max_patience = 10
    error_curr = 0
    error_prev = 0
    it = 0
    patience = 0
    approx_errors = []

    while it < max_iter and patience < max_patience:
        print "----------------------------"
        print "iteration", it
        print ""
        it_start = time.time()
        if verbose:
            t_sparse_start = time.time()
        # sparse coding
        Z = sparse_coder(X, D)
        if verbose:
            t_sparse_duration = time.time() - t_sparse_start
            print "sparse coding took", t_sparse_duration, "seconds"
            t_dict_start = time.time()

        # ksvd to learn the dictionary
        set_openblas_threads(n_jobs)
        if approx:
            D, _, unused_atoms = approx_ksvd(X, D, Z, n_cycles=n_cycles)
        elif non_neg:
            D, _, unused_atoms = nn_ksvd(X, D, Z, n_cycles=it)
        else:
            D, _, unused_atoms = ksvd(X, D, Z, n_cycles=n_cycles)
        set_openblas_threads(1)
        if verbose:
            t_dict_duration = time.time() - t_dict_start
            print "K-SVD took", t_dict_duration, "seconds"
            print ""
        if verbose:
            print "number of unused atoms:", len(unused_atoms)
        # replace the unused atoms in the dictionary
        for j in range(len(unused_atoms)):
            # no datapoint available to be used as atom
            if len(unused_data) == 0:
                break
            _idx = np.random.choice(unused_data, size=1)
            idx = _idx[0]
            D[:, unused_atoms[j]] = X[:, idx]
            D[:, unused_atoms[j]] = normalize(D[:, unused_atoms[j]])
            unused_data.remove(idx)

        if eta is not None:
            # do not force incoherence in the last iteration
            if it < max_iter - 1:
                # force Mutual Incoherence
                D, unused_data = force_mi(D, X, Z, unused_data, eta)
        if verbose:
            amc = average_mutual_coherence(D)
            print "average mutual coherence:", amc

        it_duration = time.time() - it_start
        # calculate the approximation error
        error_curr = approx_error(D, Z, X, n_jobs=2)
        approx_errors.append(error_curr)
        if verbose:
            print "error:", error_curr
            print "error difference:", (error_curr - error_prev)
            error_prev = error_curr
        print "duration:", it_duration, "seconds"
        if (it > 0) and (error_curr > 0.9 * error_prev or error_curr > error_prev):
            patience += 1
        it += 1
    print ""
    return D, Z


class ksvd_coder(object):
    """
    a wrapper to the ksvd_dict_learn function
    """

    def __init__(self, n_atoms=None, n_nonzero_coefs=None, sparse_coder=None, init_dict="data",
                 max_iter=None, non_neg=False, approx=True, eta=None, n_cycles=1, n_jobs=1,
                 mmap=False, verbose=True):
        self.n_atoms = n_atoms
        self.sparse_coder = sparse_coder
        self.max_iter = max_iter
        self.non_neg = non_neg
        self.approx = approx
        self.eta = eta
        self.n_jobs = n_jobs
        self.init_dict = init_dict
        self.n_cycles = n_cycles
        self.verbose = verbose
        self.mmap = mmap
        self.D = None

    def _fit(self, X):
        D, _ = ksvd_dict_learn(X, self.n_atoms, init_dict=self.init_dict,
                               sparse_coder=self.sparse_coder, max_iter=self.max_iter,
                               non_neg=self.non_neg, approx=self.approx, eta=self.eta, n_cycles=self.n_cycles,
                               n_jobs=self.n_jobs, mmap=self.mmap, verbose=self.verbose)
        self.D = D

    def __call__(self, X):
        self._fit(X)
        Z = self.sparse_coder(X, self.D)
        return Z

    def fit(self, X):
        self._fit(X)

    def encode(self, X):
        return self.sparse_coder(X, self.D)
