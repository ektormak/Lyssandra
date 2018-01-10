from lyssa.utils import get_empty_mmap
from .ksvd import ksvd_dict_learn
import numpy as np

"""
A module that implements class-specific dictionary learning using the KSVD algorithm. For
a dataset with C classes, it learns C dictionaries separately as in:

"Metaface learning for sparse representation based face recognition" of Yang et al.

TODO: make the class 'class_ksvd_coder' more general so that the user can specify
      any of the available Dictionary Learning algorithms.
"""


class class_ksvd_coder():
    """
    An interface to the class_dict_learn function.

    n_class_atoms and n_nonzero_coefs: arrays that specify the corresponding parameters of each class dictionary
    eta: the mutual incoherence threshold
    alpha: the structural incoherence threshold
    non_neg: if set to True, it uses non-negativity constraints
    sparse_coder: must be an instance of the sparse_coding.sparse_encoder class
    atom_ratio: if not None, it specifies the proportion of datapoints that will
                 be used as atoms in the dataset
    n_cycles: the number of KSVD cycles
    max_iter: the maximum number of KSVD iterations
    n_jobs: the number of CPU threads
    mmap: if set to True, the algorithm applies memory mapping to save memory
    """

    def __init__(self, n_class_atoms=None, n_nonzero_coefs=None,
                 atom_ratio=None, coef_ratio=None, sparse_coder=None, non_neg=False, max_iter=None,
                 approx=False, eta=None, alpha=None,
                 n_cycles=1, n_jobs=1, mmap=False, verbose=True):

        self.sparse_coder = sparse_coder
        self.n_class_atoms = n_class_atoms
        self.n_nonzero_coefs = n_nonzero_coefs
        self.eta = eta
        self.alpha = alpha
        self.non_neg = non_neg
        self.sparse_coder = sparse_coder
        self.max_iter = max_iter
        self.approx = approx
        self.atom_ratio = atom_ratio
        self.coef_ratio = coef_ratio
        self.n_cycles = n_cycles
        self.n_jobs = n_jobs
        self.mmap = mmap
        self.verbose = verbose
        self.D = None

    def _fit(self, X, y):

        n_classes = len(set(y))
        if self.n_class_atoms == None:
            self.n_class_atoms = []
            for c in range(n_classes):
                n_class_samples = np.sum(y == c)
                n_class_atoms = int(n_class_samples * self.atom_ratio)
                self.n_class_atoms.append(n_class_atoms)

        if self.n_nonzero_coefs == None:
            self.n_nonzero_coefs = []
            for c in range(n_classes):
                n_class_nonzero_coefs = int(self.n_class_atoms[c] * self.coef_ratio)
                self.n_nonzero_coefs.append(int(n_class_nonzero_coefs))

        if type(self.n_class_atoms) is not list:
            # all the class dictionaries use the same number of atoms
            self.n_class_atoms = [self.n_class_atoms for c in range(n_classes)]

        sparse_coders = [self.sparse_coder for c in range(n_classes)]

        self.D = class_dict_learn(X, y, n_class_atoms=self.n_class_atoms, sparse_coders=sparse_coders,
                                  init_dict='data', max_iter=self.max_iter, non_neg=self.non_neg,
                                  approx=self.approx, eta=self.eta,
                                  alpha=self.alpha, n_cycles=self.n_cycles,
                                  n_jobs=self.n_jobs, mmap=self.mmap, verbose=self.verbose)

    def __call__(self, X, y):
        self._fit(X, y)
        return self.D

    def fit(self, X, y):
        self._fit(X, y)

    def encode(self, X):
        Z = self.sparse_coder.encode(X, self.D)
        return Z

    def print_params(self):
        return


def class_dict_learn(X, y, n_class_atoms=None, sparse_coders=None, init_dict='data',
                     max_iter=5, approx=False, non_neg=False, eta=None, alpha=None, n_cycles=1, n_jobs=1, mmap=False,
                     verbose=True):
    n_classes = len(set(y))
    # the number of atoms in the joint dictionary
    n_total_atoms = np.sum(n_class_atoms)
    n_features = X.shape[0]
    shape = (n_features, n_total_atoms)
    if mmap:
        D = get_empty_mmap(shape)
    else:
        D = np.zeros(shape)

    for c in range(n_classes):
        if verbose:
            print "-------------------------------------"
            print "optimizing the dictionary of class", c
        x_c = y == c
        # extract the datapoints for the c-th class
        Xc = X[:, x_c]
        n_class_samples = Xc.shape[1]

        Dc, _ = ksvd_dict_learn(Xc, n_class_atoms[c], init_dict='data', sparse_coder=sparse_coders[c],
                                max_iter=max_iter, non_neg=non_neg, approx=approx, eta=eta,
                                n_cycles=n_cycles, n_jobs=n_jobs, mmap=mmap, verbose=verbose)

        base = c * n_class_atoms[c]
        offset = n_class_atoms[c]
        D[:, base:base + offset] = Dc

        if alpha is not None:
            # force Structural incoherence
            from lyssa.dict_learn.utils import replace_coherent_atoms
            if verbose:
                print "reducing structural incoherence"
            D, n_class_atoms = replace_coherent_atoms(X, y, D, n_class_atoms,
                                                      thresh=alpha, kappa=None, unused_data=None)
            return D, n_class_atoms

        # merge the class dictionaries into one
        # and return it
        return D
