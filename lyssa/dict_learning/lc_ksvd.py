from lyssa.classify import classifier
import numpy as np
from numpy.linalg import inv
from .ksvd import ksvd
from .utils import init_dictionary, approx_error
from lyssa.utils import set_openblas_threads, get_mmap
import time

"""
The LC-KSVD algorithm of "Label Consistent K-SVD: Learning A Discriminative Dictionary for Recognition"
of Jiang et al.
"""


class lc_ksvd_classifier(classifier):
    # lambda1 is the regularizer for the W parameters
    # lambda1 * ||W||_{2}
    # lambda2 is the regularizer for the transformation G parameters
    # lambda2 * ||G||_{2}
    # alpha is the weight we give for sparse code discrimination
    # beta is the weight we give for correct classification:
    # beta*||H - WZ||_{2}

    def __init__(self, class_dict_coder=None, param_grid=None,
                 sparse_coder=None, max_iter=2, approx=True, eta=0,
                 n_class_samples=None, n_test_samples=None, n_tests=1, n_folds=None,
                 alpha=1, beta=1, mmap=False, verbose=False, n_jobs=1):

        classifier.__init__(self, n_folds=n_folds, param_grid=param_grid,
                            n_class_samples=n_class_samples, n_test_samples=n_test_samples,
                            n_tests=n_tests, name='lc_ksvd_classifier')
        # n_class_atoms,n_nonzero_coefs are arrays
        # that specify the params of each class dict
        self.class_dict_coder = class_dict_coder
        self.n_class_atoms = None
        self.sparse_coder = sparse_coder
        self.max_iter = max_iter
        self.approx = approx
        # the parameters of the LC-KSVD algorithm
        self.alpha = alpha
        self.beta = beta
        self.mmap = mmap
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.sparse_coder.n_jobs = n_jobs

    def train(self, X_train, y_train, param_set=None):

        self.alpha = param_set['alpha']
        self.beta = param_set['beta']
        n_classes = len(set(y_train))
        if self.class_dict_coder is not None:
            D = self.class_dict_coder(X_train, y_train)
            self.n_class_atoms = self.class_dict_coder.n_class_atoms
        else:
            if self.n_class_atoms is None:
                # use all the datapoints in a class
                self.n_class_atoms = (np.zeros(n_classes) + self.n_class_samples).astype(int)
            n_total_atoms = np.sum(self.n_class_atoms)
            n_features = X_train.shape[0]
            D = np.zeros((n_features, n_total_atoms))
            for c in range(n_classes):
                x_c = y_train == c
                Xc = X_train[:, x_c]

                Dc = init_dictionary(Xc, self.n_class_atoms[c], method='data', normalize=True)
                base = c * self.n_class_atoms[c]
                offset = self.n_class_atoms[c]
                D[:, base:base + offset] = Dc

        print "learned class specific dictionaries...."
        n_train_samples = X_train.shape[1]
        n_atoms = np.sum(self.n_class_atoms)
        # initialize Q
        # Q_{k,i} is 1 if the ith datapoint and the k atom belong to the same class
        Q = np.zeros((n_atoms, n_train_samples))
        start = 0
        for c in range(n_classes):
            # Dc,Zc = params[k]
            yc = y_train == c
            Q[start:start + self.n_class_atoms[c], yc] = 1
            start += self.n_class_atoms[c]

        self.D, Z, self.W = lc_ksvd(X_train, y_train, D, Q, sparse_coder=self.sparse_coder,
                                    alpha=self.alpha, beta=self.beta, lambda1=1, lambda2=1,
                                    max_iter=self.max_iter, approx=self.approx, n_jobs=self.n_jobs)

    def predict(self, X_test):
        y_pred = lc_ksvd_predict(X_test, self.D, self.W, self.sparse_coder)
        return y_pred


def lc_ksvd_predict(X, D, W, sparse_coder):
    # predict the labels of the datapoints in X
    n_samples = X.shape[1]
    Z = sparse_coder(X, D)

    predictions = []
    for i in xrange(n_samples):
        pred = np.argmax(np.dot(W, Z[:, i]))
        predictions.append(pred)
    return predictions


def lc_ksvd(X, y, D, Q, alpha=1, beta=1, lambda1=1, lambda2=1,
            sparse_coder=None, max_iter=2, approx=False, mmap=False, verbose=False, n_jobs=1):
    """
    X: the data matrix with shape (n_features,n_samples)
    y: the vector that contains the label of each datapoint
    Q: a matrix with shape (n_atoms,n_samples). The element Q_{k,i} is 1 if the ith datapoint and the k atom belong to the same class
    lambda1: the regularizer for the W matrix i.e lambda1 * ||W||_{2}
    lambda2: the regularizer for the transformation matrix G i.e lambda2 * ||G||_{2}
    alpha: the weight we assign for sparse code discrimination
    beta: is the weight we assign for correct classification: beta*||H - WZ||_{2}
    """

    n_classes = len(set(y))
    n_atoms = D.shape[1]
    n_features, n_samples = X.shape
    Z = np.zeros((n_atoms, n_samples))

    # create the class label matrix
    # H is the class label matrix which has a
    # datapoint in each column with H_{c,i}=1 if
    # the ith datapoint belongs to the cth class
    H = np.zeros((n_classes, n_samples)).astype(int)

    for i in xrange(n_samples):
        H[y[i], i] = 1

    if n_jobs > 1:
        set_openblas_threads(n_jobs)
    # classifier parameter initialization
    I = np.eye(n_atoms)

    # W_{c,:} are the parameters of the linear classifier for the cth class
    W = np.dot(inv(np.dot(Z, Z.T) + lambda1 * I), np.dot(Z, H.T)).T
    # The matrix G forces the sparse codes to be discriminative and approximate the matrix Q,
    # and has shape (n_atoms,n_atoms)
    G = np.dot(inv(np.dot(Z, Z.T) + lambda2 * I), np.dot(Z, Q.T)).T

    # stack the data matrix X with class label matrix H
    # and matrix Q
    _X = np.vstack((X, np.sqrt(alpha) * Q))
    _X = np.vstack((_X, np.sqrt(beta) * H))

    if mmap:
        _X = get_mmap(_X)

    _normalizer = np.array([np.sqrt(np.dot(D[:, j], D[:, j])) for j in range(D.shape[1])])
    D = D / _normalizer
    G = G / _normalizer
    W = W / _normalizer

    # stack the dictionary D with the weight matrix W
    # and matrix G
    _D = np.vstack((D, np.sqrt(alpha) * G))
    _D = np.vstack((_D, np.sqrt(beta) * W))

    if mmap:
        _D = get_mmap(_D)

    if verbose:
        error_curr = 0
        error_prev = 0

    for it in range(max_iter):

        print "iteration", it
        it_start = time.time()
        if verbose:
            t_sparse_start = time.time()
        # sparse coding
        Z = sparse_coder(X, D)
        if verbose:
            t_sparse_duration = time.time() - t_sparse_start
            print "\nsparse coding took", t_sparse_duration, "seconds"
            t_dict_start = time.time()

        _D, _, unused_atoms = ksvd(_X, _D, Z, verbose=True)

        if verbose:
            t_dict_duration = time.time() - t_dict_start
            print "\nK-SVD took", t_dict_duration, "seconds"
        if verbose:
            print "number of unused atoms:", len(unused_atoms)

        D = _D[:n_features, :]
        G = _D[n_features:n_features + n_atoms, :]
        W = _D[n_features + n_atoms:, :]

        _normalizer = np.array([np.sqrt(np.dot(D[:, j], D[:, j])) for j in range(D.shape[1])])

        D = D / _normalizer
        G = G / _normalizer
        W = W / _normalizer
        # stack the dictionary D with the weight matrix W
        # and matrix G
        _D = np.vstack((D, np.sqrt(alpha) * G))
        _D = np.vstack((_D, np.sqrt(beta) * W))

        it_duration = time.time() - it_start
        if verbose:
            # calculate the approximation error
            error_curr = approx_error(D, Z, X, n_jobs=2)
            print "error:", error_curr
            print "error difference:", (error_curr - error_prev)
            n_correct = np.array([y[i] == np.argmax(np.dot(W, Z[:, i]))
                                  for i in range(Z.shape[1])]).nonzero()[0].size
            class_acc = n_correct / float(n_samples)
            print "classification accuracy", class_acc
            error_prev = error_curr
        print "duration:", it_duration, "seconds"
        print "----------------------"

    return D, Z, W
