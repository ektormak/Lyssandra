from lyssa.classify import linear_svm
from lyssa.utils import get_workspace
from lyssa.sparse_coding import sparse_encoder
from lyssa.dict_learning import class_ksvd_coder
from lyssa.dict_learning.lc_ksvd import lc_ksvd_classifier
from lyssa.dict_learning.src import src_classifier
from lyssa.dict_learning.utils import norm_cols


"""
Object recognition using:

1) the LC-KSVD algorithm (see http://www.umiacs.umd.edu/~lsd/papers/CVPR2011_LCKSVD_final.pdf)
2) Sparse Representation based Classification (see http://www.columbia.edu/~jw2966/papers/WYGSM09-PAMI.pdf)
3) a linear SVM
"""


def lc_ksvd_ex():

    wm = get_workspace(id=id)
    wm.show_metadata()
    X = wm.load("features.npy", online=False)
    y = wm.load("labels.npy")
    X = norm_cols(X)

    se = sparse_encoder(algorithm='bomp', params={"n_nonzero_coefs": 30}, n_jobs=8)
    ckc = class_ksvd_coder(atom_ratio=1, sparse_coder=se,non_neg=False, max_iter=5,
                           n_cycles=1, n_jobs=8, mmap=False, approx=True, verbose=True)

    lc = lc_ksvd_classifier(class_dict_coder=ckc, sparse_coder=se,
                            approx=True, max_iter=4, n_class_samples=30,n_test_samples=None,
                            verbose=True, mmap=False, n_jobs=8, param_grid=[{'alpha': [1], 'beta': [1]}])

    lc(X, y)


def ScSPM_SRC():

    wm = get_workspace(id=id)
    wm.show_metadata()
    X = wm.load("features.npy", online=False)
    y = wm.load("labels.npy")
    X = norm_cols(X)

    se = sparse_encoder(algorithm='group_omp',params={'n_groups':1},n_jobs=8)
    ckc = class_ksvd_coder(atom_ratio=1, sparse_coder=se, non_neg=False,
                           max_iter=5, n_cycles=1, n_jobs=4,
                           mmap=False, approx=True, verbose=True)

    sc = src_classifier(class_dict_coder=None, sparse_coder=se, n_class_samples=30,
                        n_test_samples=None, method="global", mmap=False, n_jobs=4)

    sc(X, y)


def multiclass_linear_svm_ex():

    wm = get_workspace(id=id)
    wm.show_metadata()
    X = wm.load("features.npy",online=False)
    y = wm.load("labels.npy")
    X = norm_cols(X)

    lsvm = linear_svm(param_grid=[{'C': [1e-1, 5e-1, 1, 5, 1e1, 5e3, 1e4]}], n_class_samples=30, n_test_samples=50)
    lsvm(X, y)

if __name__ == "__main__":
    #lc_ksvd_ex()
    #ScSPM_SRC()
    multiclass_linear_svm_ex()
