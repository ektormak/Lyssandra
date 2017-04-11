from __future__ import division
import numpy as np
from numpy.linalg import inv, solve
from numpy.linalg.linalg import LinAlgError
from lyssa.utils import fast_dot, cpu_count
from lyssa.utils.math import frobenius_squared
from scipy.linalg import solve_triangular
from lyssa.utils import run_parallel
from lyssa.utils.math import norm
from functools import partial
from scipy.optimize import nnls
from numpy.matlib import repmat

"""A module that implements Sparse Coding algorithms"""

gram_singular_msg = "Gram matrix is singular due to linear dependencies in the dictionary"


def _omp(x, D, Gram, alpha, n_nonzero_coefs=None, tol=None):
    _, n_atoms = D.shape
    # the dict indexes of the atoms this datapoint uses
    Dx = np.array([]).astype(int)
    z = np.zeros(n_atoms)
    # the residual
    r = np.copy(x)
    i = 0
    if n_nonzero_coefs is not None:
        tol = 1e-10
        def cont_criterion():
            not_reached_sparsity = i < n_nonzero_coefs
            return (not_reached_sparsity and norm(r) > tol)
    else:
        cont_criterion = lambda: norm(r) >= tol

    while (cont_criterion()):

        # find the atom that correlates the
        # most with the residual
        k = np.argmax(np.abs(alpha))
        if k in Dx:
            break
        Dx = np.append(Dx, k)
        # solve the Least Squares problem
        # to find the coefs z
        DI = D[:, Dx]
        G = Gram[Dx, :][:, Dx]
        G = np.atleast_2d(G)
        try:
            G_inv = inv(G)
        except LinAlgError:
            print gram_singular_msg
            break

        z[Dx] = np.dot(G_inv, np.dot(D.T, x)[Dx])
        r = x - np.dot(D[:, Dx], z[Dx])
        alpha = np.dot(D.T, r)
        i += 1

    return z


def omp(X, Alpha, D, Gram, n_nonzero_coefs=None, tol=None):
    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = np.zeros((n_atoms, n_samples))
    for i in range(n_samples):
        Z[:, i] = _omp(X[:, i], D, Gram, Alpha[:, i], n_nonzero_coefs=n_nonzero_coefs, tol=tol)
    return Z


def group_omp(X, Alpha, D, Gram, groups=None, n_groups=None, tol=None):
    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = np.zeros((n_atoms, n_samples))
    for i in xrange(n_samples):
        Z[:, i] = _group_omp(X[:, i], D, Gram, Alpha[:, i], groups=groups, n_groups=n_groups, tol=tol)
    return Z


def _group_omp(x, D, Gram, alpha, groups=None, n_groups=None, tol=None):
    # TODO: also use a tolerance parameter
    _, n_atoms = D.shape
    # the dict indexes of the groups
    # this datapoint uses
    Gx = np.array([]).astype(int)
    z = np.zeros(n_atoms)
    # the residual
    r = np.copy(x)
    i = 0
    if n_groups is not None:
        tol = 1e-10

        def cont_criterion():
            not_reached_sparsity = i < n_groups
            return (not_reached_sparsity and norm(r) > tol)
    else:
        cont_criterion = lambda: norm(r) > tol

    while (cont_criterion()):
        # find the group of atoms that correlates
        # the most with the residual
        if i == 0:
            group_scores = [norm(alpha[group]) for group in groups]
        else:
            group_scores = [norm(np.dot(D[:, group].T, r)) for group in groups]
        g = np.argmax(group_scores)
        if g in Gx or norm(r) < 1e-10:
            # group already selected
            break
        Gx = np.append(Gx, g)
        # solve the Least Squares problem
        # to find the coefs z
        idx = np.array([k for g_idx in Gx for k in groups[g_idx]])
        G = Gram[idx, :][:, idx]

        try:
            G_inv = inv(G)
        except LinAlgError:
            print gram_singular_msg
            break

        z[idx] = np.dot(np.dot(G_inv, D[:, idx].T), x)
        approx = np.dot(D[:, idx], z[idx])
        r = x - approx
        i += 1
    return z


def sparse_group_omp(X, D, Gram, groups=None, n_groups=None, n_nonzero_coefs=None):
    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = np.zeros((n_atoms, n_samples))
    Alpha = fast_dot(D.T, X)
    for i in xrange(n_samples):
        Z[:, i] = _sparse_group_omp(X[:, i], D, Gram, Alpha[:, i], groups=groups, n_groups=n_groups,
                                    n_nonzero_coefs=n_nonzero_coefs)
    return Z


def _sparse_group_omp(x, D, Gram, alpha, groups=None, n_groups=None, n_nonzero_coefs=None):

    _, n_atoms = D.shape
    # the dict indexes of the groups
    # this datapoint uses
    Gx = np.array([])
    Gx = Gx.astype(int)
    z = np.zeros(n_atoms)
    # the residual
    r = np.copy(x)

    i = 0
    if n_groups is not None:
        tol = 1e-10

        def cont_criterion():
            not_reached_sparsity = i < n_groups
            return (not_reached_sparsity and norm(r) > tol)
    else:
        cont_criterion = lambda: norm(r) > tol

    # first select the groups
    for i in range(n_groups):
        # find the group of atoms that correlates the
        # most with the residual
        if i == 0:
            group_scores = [norm(alpha[group]) for group in groups]
        else:
            group_scores = [norm(np.dot(D[:, group].T, r)) for group in groups]
        g = np.argmax(group_scores)
        if g in Gx or norm(r) < 1e-10:
            # group already selected
            break
        Gx = np.append(Gx, g)
        # solve the Least Squares problem
        # to find the coefs z
        idx = np.array([k for g_idx in Gx for k in groups[g_idx]])
        G = Gram[idx, :][:, idx]

        try:
            G_inv = inv(G)
        except LinAlgError:
            print gram_singular_msg
            break

        z[idx] = np.dot(np.dot(G_inv, D[:, idx].T), x)
        approx = np.dot(D[:, idx], z[idx])
        r = x - approx
        i += 1

    # apply OMP using only the atoms of the groups selected
    Dx = np.array([])
    Dx = Dx.astype(int)
    # the atom indices selected from the previous step
    idx = np.array([k for g_idx in Gx for k in groups[g_idx]])
    Dsel = D[:, idx]
    Gram = fast_dot(Dsel.T, Dsel)
    z = np.zeros(len(idx))
    z_final = np.zeros(n_atoms)
    # the residual
    r = np.copy(x)

    i = 0
    if n_nonzero_coefs is not None:
        tol = 1e-20

        def cont_criterion():
            not_reached_sparsity = i < n_nonzero_coefs
            return (not_reached_sparsity and norm(r) > tol)
    else:
        cont_criterion = lambda: norm(r) > tol

    while (cont_criterion()):

        # find the atom that correlates the
        # most with the residual
        k = np.argmax(np.abs(np.dot(Dsel.T, r)))
        if k in Dx:
            break
        Dx = np.append(Dx, k)
        # solve the Least Squares problem
        # to find the coefs z
        DI = Dsel[:, Dx]
        G = Gram[Dx, :][:, Dx]
        G = np.atleast_2d(G)
        try:
            G_inv = inv(G)
        except LinAlgError:
            print gram_singular_msg
            break

        z[Dx] = np.dot(G_inv, np.dot(Dsel.T, x)[Dx])
        z_final[idx[Dx]] = z[Dx]
        r = x - np.dot(Dsel[:, Dx], z[Dx])

        i += 1

    return z_final


def somp(X, D, Gram, data_groups=None, n_nonzero_coefs=None):
    # the Simultaneous OMP algorirthm
    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    n_groups = len(data_groups)
    Z = np.zeros((n_atoms, n_samples))
    for g in range(n_groups):
        Z[:, data_groups[g]] = _somp(X[:, data_groups[g]], D, Gram, n_nonzero_coefs=n_nonzero_coefs)
    return Z


def _somp(X_g, D, Gram, n_nonzero_coefs=None):
    n_atoms = D.shape[1]
    n_group_samples = X_g.shape[1]
    Z = np.zeros((n_atoms, n_group_samples))
    Dx = np.array([])
    Dx = Dx.astype(int)
    R = X_g

    if n_nonzero_coefs is not None:
        tol = 1e-20

        def cont_criterion():
            not_reached_sparsity = i < n_nonzero_coefs
            return (not_reached_sparsity and frobenius_squared(R) > tol)
    else:
        cont_criterion = lambda: frobenius_squared(R) > tol

    i = 0
    while (cont_criterion()):

        A = fast_dot(D.T, R)
        j = np.argmax([norm(A[k, :]) for k in range(n_atoms)])

        Dx = np.append(Dx, j)
        G = Gram[Dx, :][:, Dx]
        G = np.atleast_2d(G)
        try:
            G_inv = inv(G)
        except LinAlgError:
            print gram_singular_msg
            break

        Z[Dx, :] = fast_dot(fast_dot(inv(G_inv), D[:, Dx].T), X_g)
        R = X_g - fast_dot(D, Z)
        i += 1

    return Z


def ormp(X, D, n_nonzero_coefs=None, tol=None, n_jobs=1):
    # Order Recursive Matching Pursuit implementation of SPAMS package as introduced in
    # "Forward Sequential Algorithms for Best Basis Selection"

    import spams
    if n_nonzero_coefs is not None:
        Z = spams.omp(np.asfortranarray(X), np.asfortranarray(D), L=n_nonzero_coefs,
                      return_reg_path=False, numThreads=n_jobs)
    elif tol is not None:
        Z = spams.omp(np.asfortranarray(X), np.asfortranarray(D), eps=tol,
                      return_reg_path=False, numThreads=n_jobs)
    return np.array(Z.todense())


def batch_omp(X, Alpha, D, Gram, n_nonzero_coefs=None, tol=None):
    # applies sparsity constraint batch_omp to each datapoint in
    # a column of X

    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = np.zeros((n_atoms, n_samples))

    for i in range(n_samples):
        # the indexes of the atoms this datapoint uses
        Dx = np.array([]).astype(int)
        G = Gram
        a0 = Alpha[:, i]
        a = a0
        _eps = np.finfo(float).eps
        L = np.zeros((n_nonzero_coefs, n_nonzero_coefs))

        for j in xrange(n_nonzero_coefs):
            # find the atom that correlates the
            # most with the residual
            k = np.argmax(np.abs(a))
            if k in Dx:
                # atom already selected
                break

            g = G[Dx, k]
            # after first iteration
            if j > 0:
                if j == 1:
                    # second iteration
                    w = g
                    v = w * w
                    vs = 1 - v
                    if vs < _eps:
                        break
                    L[:2, :2] = [[1, 0],
                                 [w, np.sqrt(vs)]]

                else:
                    # after second iteration
                    w = solve_triangular(L[:j, :j], g, lower=True, check_finite=False)
                    v = np.dot(w, w)
                    vs = 1 - v
                    if vs < _eps:
                        break

                    L[j, :j] = w
                    L[j, j] = np.sqrt(vs)
                # solve for z
                Dx = np.append(Dx, k)
                try:
                    Ltc = solve_triangular(L[:j + 1, :j + 1], a0[Dx], lower=True)
                    z = solve_triangular(L[:j + 1, :j + 1], Ltc, trans=1, lower=True)
                except LinAlgError:
                    print gram_singular_msg
                    Dx = Dx[:-1]
                    break
                a = a0 - np.dot(G[:, Dx], z)
            else:
                Dx = np.append(Dx, k)
                z = a0[Dx]
                a = a0 - np.dot(G[:, Dx], z)

        Z[Dx, i] = z

    return Z



def nn_omp(X, D, n_nonzero_coefs=None, tol=None):
    """ The Non Negative OMP algorithm of
        'On the Uniqueness of Nonnegative Sparse Solutions to Underdetermined Systems of Equations'"""

    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    Z = np.zeros((n_atoms, n_samples))
    _norm = np.sum(D ** 2, axis=0)
    for i in range(n_samples):

        x = X[:, i]
        r = x
        z = np.zeros(n_atoms)
        Dx = np.array([]).astype(int)
        j = 0
        if n_nonzero_coefs is not None:
            tol = 1e-20

            def cont_criterion():
                not_reached_sparsity = j < n_nonzero_coefs
                return (not_reached_sparsity and norm(r) > tol)
        else:
            cont_criterion = lambda: norm(r) > tol

        while (cont_criterion()):
            a = np.dot(D.T, r)
            a[a < 0] = 0
            e = (norm(r) ** 2) - (a ** 2) / _norm
            k = np.argmin(e)
            Dx = np.append(Dx, k)

            z_est = nnls(D[:, Dx], x)[0]
            r = x - np.dot(D[:, Dx], z_est)
            j += 1

        if j != 0:
            z[Dx] = z_est
        Z[:, i] = z
    return Z


def soft_thresh(z, _lambda):
    # the soft thresholding operator
    return np.array([np.sign(z[k]) * max(np.abs(z[k]) - _lambda, 0) for k in xrange(z.size)])


def thresholding(Alpha, nonzero_percentage=None, n_nonzero_coefs=None):
    n_atoms, n_samples = Alpha.shape
    Z = np.zeros((n_atoms, n_samples))
    if nonzero_percentage is not None:
        n_nonzero_coefs = int(np.floor(nonzero_percentage * n_atoms))

    for i in xrange(n_samples):
        idx = Alpha[:, i].argsort()[::-1][:n_nonzero_coefs]
        Z[idx, i] = Alpha[idx, i]
    return Z


def _grad_desc_step(x, z, D, learning_rate):
    """ performs one step of gradient descent with respect to the sparse codes while fixing D"""
    return z + learning_rate * (np.dot(D.T, x - np.dot(D, z)))


def iterative_hard_thresh(X, Z0, R0, D, eta=None, n_nonzero_coefs=None, n_iter=None):
    n_samples = X.shape[1]
    Z = Z0
    R = R0
    for it in xrange(n_iter):

        Z -= eta * np.dot(D.T, R)
        for i in xrange(n_samples):
            # zero out all the entries
            # that have small values
            idx = np.abs(Z[:, i]).argsort()[::-1][n_nonzero_coefs:]
            Z[idx, i] = 0
        R = np.dot(D, Z) - X
    return Z


def llc(X, D, knn=5):
    # the sparse coder introduced in
    # "Locality-constrained Linear Coding for Image Classification"

    n_samples = X.shape[1]
    n_atoms = D.shape[1]
    # has the distance of
    # each sample to each atom
    dist = np.zeros((n_samples, n_atoms))
    # calculate the distances
    for i in range(n_samples):
        for j in range(n_atoms):
            dist[i, j] = norm(X[:, i] - D[:, j])

    # has the indices of the atoms
    # that are nearest neighbour to each sample
    knn_idx = np.zeros((n_samples, knn)).astype(int)
    for i in xrange(n_samples):
        knn_idx[i, :] = np.argsort(dist[i, :])[:knn]
    # the sparse coding matrix
    Z = np.zeros((n_atoms, n_samples))
    II = np.eye(knn)
    beta = 1e-4
    b = np.ones(knn)
    for i in xrange(n_samples):
        idx = knn_idx[i, :]
        z = D.T[idx, :] - repmat(X.T[i, :], knn, 1)
        C = np.dot(z, z.T)
        C = C + II * beta * np.trace(C)
        # solve the linear system C*c=b
        c = solve(C, b)
        # enforce the constraint on the sparse codes
        # such that sum(c)=1
        c = c / float(np.sum(c))
        Z[idx, i] = c

    return Z


class lasso():
    """
    a minimal wrapper of the spams.lasso solver
    """

    def __init__(self, _lambda, n_jobs):
        self._lambda = _lambda
        self.n_jobs = n_jobs

    def encode(self, X, D):
        return self.__call__(X, D)

    def __call__(self, X, D):
        import spams
        lasso_params = {
            'lambda1': self._lambda,
            'lambda2': 0,
            'numThreads': self.n_jobs,
            'mode': 2}

        return np.array(spams.lasso(np.asfortranarray(X, np.float64), D=np.asfortranarray(D, np.float64),
                                    return_reg_path=False, **lasso_params).todense())


class sparse_encoder(object):
    """
    A class that interfaces the functions defined above.
    The user must specify the Sparse Coding algorithm and it's
    parameters in the param dictionary.

    algorithm can be one of the following:

    'omp' => Orthognal Matching Pursuit with Least Sqaures

             params:
                    n_nonzero_coefs: the number of non-zero coefficients
                                     of the sparse representations (i.e sparsity)

                    tol: the error bound that should be achieved
                         in the approximation


    'bomp' => Batch Orthognal Matching Pursuit algorithm

             params:
                    n_nonzero_coefs: the number of non-zero coefficients
                                     of the sparse representations (i.e sparsity)

                    tol: to be implemented

    'nnomp' => Non-Negative Orthogonal Matching Pursuit algorithm. Solves the
               l0 problem like 'omp' and 'bomp' but enforce the solutions to
               be non-negative vectors.

               params: (same as 'omp' and 'bomp')




    'iht' => Iterative Hard Thresholding

             params:
                    learning_rate: the learning rate of the gradient procedure

                    n_iter: the number of iterations

                    threshold: the threshold of the hard thresholding operator


    'lasso' => Least Absolute Shrinkage and Selection operator

               params:
                       lambda: the l1 penalty parameter


    'somp' => Simultaneous Orthogonal Matching Pursuit. It jointly encodes signals
               of the same group.

              params:

                     data_groups:  a list of the datapoint indices
                                   that belong to the same group

                     n_nonzero_coefs: the number of non-zero coefficients
                                      of the sparse representations (i.e sparsity)

    'group_omp' => sparsity constraint Group Orthognal Matching Pursuit as described in
                     "Aurelie C. Lozano, Grzegorz Swirszcz, Naoki Abe:  Group Orthogonal Matching Pursuit for
                   Variable Selection and Prediction"

                   params:

                          groups:   a list of the atom indices
                                      that belong to the same group

                          n_groups  the number of atom groups to be selected
                                      per atom
    """

    def __init__(self, algorithm='omp', params=None, n_jobs=1, verbose=True, mmap=False, name='sparse_coder'):
        self.name = name

        self.algorithm = algorithm
        self.params = params
        if self.params is None:
            self.params = {}
        if n_jobs == -1:
            n_jobs = cpu_count
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.mmap = mmap

    def encode(self, X, D):
        return self.__call__(X, D)

    def __call__(self, X, D):
        # assume X has datapoints in columns
        # use self.params.get('key') because it does not throw exception
        # when the key does not exist, it just returns None.
        from lyssa.utils import set_openblas_threads

        n_samples = X.shape[1]
        n_atoms = D.shape[1]
        n_batches = 100

        if self.params.get('lambda') is not None:
            assert self.params.get('lambda') <= n_atoms

        if self.n_jobs > 1:
            set_openblas_threads(self.n_jobs)

        batched_args = None

        if self.algorithm == 'omp':

            Gram = fast_dot(D.T, D)
            args = [D, Gram]
            Alpha = fast_dot(D.T, X)
            batched_args = [Alpha]
            data = X
            func = partial(omp, n_nonzero_coefs=self.params.get('n_nonzero_coefs'), tol=self.params.get('tol'))

        elif self.algorithm == 'bomp':

            Gram = fast_dot(D.T, D)
            Alpha = fast_dot(D.T, X)
            batched_args = [Alpha]
            args = [D, Gram]
            data = X
            func = partial(batch_omp, n_nonzero_coefs=self.params.get('n_nonzero_coefs'), tol=self.params.get('tol'))

        elif self.algorithm == 'thresh':
            Alpha = fast_dot(D.T, X)
            data = Alpha
            args = []
            func = partial(thresholding, n_nonzero_coefs=self.params.get('n_nonzero_coefs'),
                           nonzero_percentage=self.params.get('nonzero_percentage'))

        elif self.algorithm == "nnomp":

            args = [D]
            data = X
            func = partial(nn_omp, n_nonzero_coefs=self.params.get('n_nonzero_coefs'), tol=self.params.get('tol'))

        elif self.algorithm == 'group_omp':

            Gram = fast_dot(D.T, D)
            Alpha = fast_dot(D.T, X)
            batched_args = [Alpha]
            data = X
            func = partial(group_omp, groups=self.params.get('groups'), n_groups=self.params.get('n_groups'))
            args = [D, Gram]

        elif self.algorithm == 'sparse_group_omp':

            # group_omp(X,D,Gram,groups=None,n_groups=None)
            Gram = fast_dot(D.T, D)
            data = X
            # sparse_group_omp(X,D,Gram,groups=None,n_groups=None,n_nonzero_coefs=None)
            func = partial(sparse_group_omp, groups=self.params.get('groups'), n_groups=self.params.get('n_groups'),
                           n_nonzero_coefs=self.params.get('n_nonzero_coefs'))
            args = [D, Gram]

        elif self.algorithm == 'somp':
            Gram = fast_dot(D.T, D)
            data = X
            func = partial(somp, data_groups=self.params.get('data_groups'),
                           n_nonzero_coefs=self.params.get('n_nonzero_coefs'))
            args = [D, Gram]

        elif self.algorithm == 'iht':

            Alpha = fast_dot(D.T, X)
            data = Alpha
            args = []
            func = partial(thresholding, n_nonzero_coefs=self.params.get('n_nonzero_coefs'),
                           nonzero_percentage=self.params.get('nonzero_percentage'))

            Z0 = run_parallel(func=func, data=data, args=args, batched_args=batched_args,
                              result_shape=(n_atoms, n_samples), n_batches=n_batches,
                              mmap=self.mmap, n_jobs=self.n_jobs)

            R0 = fast_dot(D, Z0) - X
            data = X
            batched_args = [Z0, R0]
            args = [D]
            # iterative_hard_thresh(X,Z0,Alpha,D,eta=None,n_nonzero_coefs=None,n_iter=None)
            func = partial(iterative_hard_thresh, n_nonzero_coefs=self.params.get('n_nonzero_coefs'),
                           eta=self.params.get('eta'), n_iter=self.params.get('n_iter'))
            """params = sparse_coder['iterative_hard_thresh']
            learning_rate = params[0]
            threshold = params[1]
            max_iter = params[2]
            Z = iterative_hard_thresh(X,D,Z,learning_rate=learning_rate,threshold = threshold,max_iter = max_iter)
            """

        elif self.algorithm == 'lasso':
            return lasso(self.params.get('lambda'), self.n_jobs)(X, D)

        elif self.algorithm == 'llc':

            func = partial(llc, knn=self.params.get('knn'))
            data = X
            args = [D]

        if self.verbose:
            msg = "sparse coding"
        else:
            msg = None

        if self.n_jobs > 1:
            # disable openblas to
            # avoid the hanging problem
            set_openblas_threads(1)

        Z = run_parallel(func=func, data=data, args=args, batched_args=batched_args,
                         result_shape=(n_atoms, n_samples), n_batches=n_batches,
                         mmap=self.mmap, msg=msg, n_jobs=self.n_jobs)

        # restore the previous setting
        if self.n_jobs > 1:
            set_openblas_threads(self.n_jobs)

        return Z
