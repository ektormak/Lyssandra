import numpy as np
from lyssa.utils.math import fast_dot
from lyssa.classify import classifier
from lyssa.utils import norm_cols, set_openblas_threads, run_parallel
from lyssa.dict_learning.utils import get_class_atoms, approx_error, init_dictionary

"""
A module that implements the classification system of
"Robust Face Recognition via Sparse Representation" by Wright et al.

Note:
the difference between global and local SRC classifiers is explained in the paper
"A Dictionary Learning Approach for Classification: Seperating the Particularity and the Commonality"
"""


def sci(z, n_class_atoms, reg='l0'):
    """
    the sparsity concentration index of a coefficient vector z
    for a given regularizer
    """

    n_classes = len(n_class_atoms)
    if reg == 'l0':
        norm_const = z.nonzero()[0].size
        nom = np.max([z[get_class_atoms(c, n_class_atoms)].nonzero()[0].size for c in range(n_classes)])
    elif reg == 'l1':
        nom = np.max([np.sum(np.abs(z[get_class_atoms(c, n_class_atoms)])) for c in range(n_classes)])
        norm_const = np.sum(np.abs(z))

    sci_index = (n_classes * (nom / norm_const) - 1) / float(n_classes - 1)
    return sci_index


def global_error(X, D, sparse_coder, n_class_atoms, n_jobs=1):
    """
    computes the approximation error of the dataset to
    each class-specific dictionary. The dataset is first encoded over the
    joint dictionary.
    """

    Z = sparse_coder(X, D)
    n_samples = X.shape[1]
    n_classes = len(n_class_atoms)
    E = np.zeros((n_classes, n_samples))

    if n_jobs > 1:
        set_openblas_threads(n_jobs)

    for c in range(n_classes):
        c_idx = get_class_atoms(c, n_class_atoms)
        E[c, :] = np.sum(np.power(fast_dot(D[:, c_idx], Z[c_idx, :]) - X, 2), axis=0)

    if n_jobs > 1:
        set_openblas_threads(1)

    return E


def local_error(X, D, n_class_atoms, sparse_coder):
    """
    computes the approximation error of the dataset to
    each class-specific dictionary. Each datapoint is first encoded over the
    each dictionary seperately.
    """
    n_samples = X.shape[1]
    n_classes = len(n_class_atoms)
    E = np.zeros((n_classes, n_samples))
    for c in range(n_classes):
        c_idx = get_class_atoms(c, n_class_atoms)
        Dc = D[:, c_idx]
        Zc = sparse_coder(X, Dc)
        E[c, :] = approx_error(Dc, Zc, X)
    return E


def global_error_predict(X, D, sparse_coder, n_class_atoms, n_jobs=1):
    """
    predict the labels of the datapoints in X
    using a global SRC classifier. In other words, we first encode
    each datapoint over the joint dictionary D = [D_{1} | ... | D_{C}]
    and then we assign the datapoint to the class that its dictionary
    achieves the best approximation.
    """
    E = global_error(X, D, sparse_coder, n_class_atoms, n_jobs=1)
    n_samples = E.shape[1]
    predictions = []
    for i in xrange(n_samples):
        pred = np.argmin(E[:, i])
        predictions.append(pred)

    return predictions


def global_sparse_predict(X, D, sparse_coder, n_class_atoms):
    """
    works in the same way as 'global_error_predict' but
    we classify according to the sparsest solution
    """

    Z = sparse_coder(X, D)
    n_samples = X.shape[1]
    n_classes = len(n_class_atoms)
    predictions = []

    for i in xrange(n_samples):
        sp = np.zeros((n_classes)).astype(int)

        for c in range(n_classes):
            c_idx = get_class_atoms(c, n_class_atoms)
            sp[c] = Z[c_idx, i].nonzero()[0].size

        pred = np.argmin(sp)
        predictions.append(pred)

    return predictions


def local_error_predict(X, D, sparse_coder, n_class_atoms):
    """
    predict the labels of the datapoints in X
    using a local SRC classifier. That is, we encode
    each datapoint in each class-specific dictionary
    seperately.
    """

    n_samples = X.shape[1]
    E = local_error(X, D, n_class_atoms, sparse_coder)

    predictions = []
    for i in xrange(n_samples):
        pred = np.argmin(E[:, i])
        predictions.append(pred)
    return predictions


def local_sparse_predict(X, D, sparse_coder, n_class_atoms):
    n_samples = X.shape[1]
    n_classes = len(n_class_atoms)
    n_total_atoms = np.sum(n_class_atoms)
    # sparsely encode each datapoint over each class specific dictionary
    Z = np.zeros((n_total_atoms, n_samples))
    for c in range(n_classes):
        c_idx = get_class_atoms(c, n_class_atoms)
        Zc = sparse_coder(X, D[:, c_idx])
        Z[c_idx, :] = Zc

    predictions = []
    for i in xrange(n_samples):

        sp = np.zeros((n_classes)).astype(int)
        for c in range(n_classes):
            c_idx = get_class_atoms(c, D, n_class_atoms)
            sp[c] = Z[c_idx, i].nonzero()[0].size

        pred = np.argmin(sp)
        predictions.append(pred)

    return predictions


def global_src_features(X, D, sparse_coder, n_class_atoms, n_jobs=1):
    """
    return the features for each datapoint which
    are the approximation errors of the datapoint encoded
    over each sub-dictionary in D
    """
    E = global_error(X, D, sparse_coder, n_class_atoms, n_jobs=n_jobs)
    Z_final = norm_cols(E)
    return Z_final


def local_src_features(X, D, sparse_coder, n_class_atoms, n_jobs=1):
    n_samples = X.shape[1]
    n_classes = len(n_class_atoms)
    data = [X]
    args = [D, n_class_atoms, sparse_coder]
    Z_final = run_parallel(func=local_error, data=data, args=args, batched_args=None,
                           result_shape=(n_classes, n_samples), n_batches=100, mmap=False,
                           msg="building global SRC features", n_jobs=n_jobs)

    Z_final = norm_cols(Z_final)
    return Z_final


def src_predict(X, D, n_class_atoms, sparse_coder, method="global", n_jobs=1):
    """
    D contains the sub-dictionaries, i.e
    D = [D1,...,Dc]

    if n_nonzero_coefs is None then we solve the error constrained
    problem using the tol parameter and we predict according to the
    sparsest solution

    if tol is None then we solve the sparsity constrained
    problem using the n_non_zero_coefs parameter and we predict according to the
    lowerst approximation error
    """

    solve_sparse = solve_error = False
    if 'n_nonzero_coefs' or 'n_groups' in sparse_coder.params.keys():
        solve_sparse = True
    elif 'tol' in sparse_coder.params.keys():
        solve_error = True

    if 'n_groups' in sparse_coder.params.keys():
        sparse_coder.params["groups"] = [range(c * cl_atoms, (c + 1) * cl_atoms) for c, cl_atoms in
                                         enumerate(n_class_atoms)]

    predictions = None
    if solve_sparse and method == "global":
        predictions = global_error_predict(X, D, sparse_coder, n_class_atoms, n_jobs=n_jobs)
    elif solve_error and method == "global":
        predictions = global_sparse_predict(X, D, sparse_coder, n_class_atoms)
    elif solve_sparse and method == "local":
        predictions = local_error_predict(X, D, sparse_coder, n_class_atoms)
    elif solve_error and method == "local":
        predictions = local_sparse_predict(X, D, sparse_coder, n_class_atoms)

    return predictions


class src_classifier(classifier):
    def __init__(self, class_dict_coder=None, n_folds=None,
                 sparse_coder=None,
                 n_class_samples=None, n_test_samples=None, n_tests=1,
                 method="global", mmap=False, n_jobs=1):

        classifier.__init__(self, n_folds=n_folds,
                            n_class_samples=n_class_samples, n_test_samples=n_test_samples,
                            n_tests=n_tests, name=method + '_src_classifier')

        self.class_dict_coder = class_dict_coder
        self.sparse_coder = sparse_coder
        # if method='global' then we apply the global SRC classifier
        # if method='local' then we apply the local SRC classifier
        self.method = method
        self.mmap = mmap
        self.n_jobs = n_jobs

    def train(self, X_train, y_train, param_set=None):

        n_classes = len(set(y_train))

        if self.class_dict_coder is not None:
            # ksvd in each class dictionary
            self.class_dict_coder.mmap = self.mmap
            self.class_dict_coder.n_jobs = self.n_jobs
            self.D = self.class_dict_coder(X_train, y_train)
            self.n_class_atoms = self.class_dict_coder.n_class_atoms
        else:
            # every datapoint is an atom
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

            self.D = D

    def predict(self, X_test):
        self.sparse_coder.mmap = self.mmap
        self.sparse_coder.n_jobs = self.n_jobs
        y_pred = src_predict(X_test, self.D, self.n_class_atoms, self.sparse_coder,
                             method=self.method, n_jobs=self.n_jobs)

        return y_pred


class src_feature_classifier(classifier):
    def __init__(self, class_dict_coder=None, n_folds=None,
                 sparse_coder=None, param_grid=None,
                 n_class_samples=None, n_test_samples=None, n_tests=1,
                 method="global", mmap=False, n_jobs=1):

        classifier.__init__(self, n_folds=n_folds, param_grid=param_grid,
                            n_class_samples=n_class_samples, n_test_samples=n_test_samples,
                            n_tests=n_tests, name=method + 'src_feature_classifier')

        # a class that will do class dictionary learning
        # of the data
        self.class_dict_coder = class_dict_coder
        self.sparse_coder = sparse_coder
        # if method='global' then we extract global SRC features
        # if method='local' then we extract local SRC features
        self.method = method
        self.mmap = mmap
        self.n_jobs = n_jobs
        self.sparse_coder.mmap = self.mmap
        self.sparse_coder.n_jobs = self.n_jobs
        self.D = None
        self.features_extracted = False
        self.Z_train = None
        self.Z_test = None

    def train(self, X_train, y_train, param_set=None):

        n_classes = len(set(y_train))

        if self.D is None:
            if self.class_dict_coder is not None:
                # ksvd in each class dictionary
                self.class_dict_coder.mmap = self.mmap
                self.class_dict_coder.n_jobs = self.n_jobs
                self.D = self.class_dict_coder(X_train, y_train)
                self.n_class_atoms = self.class_dict_coder.n_class_atoms
            else:
                # every datapoint is an atom
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

                self.D = D

        from lyssa.classify import linear_svm
        self.lsvm = linear_svm()

        # if not self.features_extracted:
        if self.method == "global":
            self.Z_train = global_src_features(X_train, self.D, self.sparse_coder, self.n_class_atoms,
                                               n_jobs=self.n_jobs)
        elif self.method == "local":
            pass
        # self.features_extracted = True
        self.lsvm.train(self.Z_train, y_train, param_set=param_set)

    def predict(self, X_test):
        # if not self.features_extracted:
        if self.method == "global":
            self.Z_test = global_src_features(X_test, self.D, self.sparse_coder, self.n_class_atoms, n_jobs=self.n_jobs)
        elif self.method == "local":
            pass
        # self.features_extracted = True
        y_pred = self.lsvm.predict(self.Z_test)

        return y_pred


"""
def atom_quality(Z,y,sparse_coder,n_class_atoms,mode=0):

    print "measuring atom quality"
    #idea1: find the fraction of the datapoints that use each atom
    # and are in the same class with the total number of datapoints that
    # use this atom
    #idea2: for each atom contruct a weighted average using the datapoints
    # that belong to a class different than it. The weights are equal
    # to the absolute value of the coefficients these datapoints have
    # for this atom
    n_classes = len(n_class_atoms)
    atom_scores = np.zeros(np.sum(n_class_atoms))
    for c in range(n_classes):

        class_idx = np.where(y==c)[0]
        atoms_idx = get_class_atoms(c,n_class_atoms)
        #for each atom in this class
        for atom_idx in atoms_idx:
            #find the indices of the datapoints that
            #use this atom
            data_idx = Z[atom_idx,:].nonzero()[0]
            if mode == 0:
                if len(data_idx) == 0:
                    #this atom is not used
                    continue
                #find the number of datapoints that use this
                #atom and are in the same class
                n_correct = len(set(data_idx).intersection(class_idx))
                n_total = len(data_idx)
                #print atom_idx
                #idea1:
                atom_score = n_correct / float(n_total)
            elif mode == 1:

                #idea2:
                #find all the datapoints that use this atom but belong to
                #a different class
                diff_idx = set(data_idx).difference(set(class_idx))
                if len(diff_idx) == 0:
                    continue
                #find the coeffs in absolute value
                atom_score = - float( np.sum(np.abs(Z[atom_idx,list(diff_idx) ])) )

            atom_scores[atom_idx] = atom_score
    return atom_scores
"""

# extracts features in this way:
# for each datapoint:
# extract the feature vector [e_{1},...,e_{C}]
# where e_{c} is the approximation error of the datapoint
# coded over the cth class dictionary
# it can be based on local or global SRC
"""
class src_feature_extractor(classifier):

    def __init__(self,class_dict_coder=None,n_nonzero_coefs=None,tol=None
                ,approx=True,n_folds=None,n_class_samples=None,n_test_samples=None,n_tests=1,
                method="global",mmap=False,n_jobs=1):


        classifier.__init__(self,n_folds=n_folds,
                n_class_samples=n_class_samples,n_test_samples=n_test_samples,
                n_tests=n_tests,name='src_feature_classifier')


        #a class that will do class dictionary learning
        #of the data
        self.class_dict_coder = class_dict_coder
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.approx = approx
        #if method='global' then we apply the global SRC classifier
        #if method='local' then we apply the local SRC classifier
        self.method = method
        self.mmap = mmap
        self.class_dict_coder.mmap = self.mmap
        self.n_jobs = n_jobs



    def train(self,X_train,y_train):
        '''train the classifier'''

        n_classes = len(set(y_train))
        if self.n_nonzero_coefs is not None:
            sparse_coder = sparse_encoder(algorithm='ormp',params={'n_nonzero_coefs':self.n_nonzero_coefs},
                                                mmap=self.mmap,n_jobs=self.n_jobs)
        elif self.tol is not None:
            sparse_coder = sparse_encoder(algorithm='ormp',params={'tol':self.tol},
                                                mmap=self.mmap,n_jobs=self.n_jobs)

        self.param_grid = [ {'C': [1e-3,1e-2,5e-2,1e-1,1,10,50]} ]


        #THE TRAINING:
        #learn a dictionary per class using ksvd
        #and merge the subdictionaries to D
        D = self.class_dict_coder(X_train,y_train)
        n_class_atoms = self.class_dict_coder.n_class_atoms

        if self.method == "global":
            Z = global_src_features(X,D,y_train,sparse_coder,
                n_class_atoms,train_idx,test_idx,n_jobs=self.n_jobs)
        elif self.method == "local":
            Z = local_src_features(X,D,sparse_coder,n_class_atoms,n_jobs=self.n_jobs)

        Z_train = Z[:,train_idx]
        Z_test = Z[:,test_idx]

        self.clf.fit(Z_train.T,y_train)



    def predict(self,X_test):
        '''test the classifier'''
        y_pred = clf.predict(Z_test.T)

        return y_pred


class supervised_src_classifier(classifier):

    def __init__(self,class_dict_coder=None,n_nonzero_coefs=None,tol=None,n_folds=None,
                sparse_coder=None,
                n_class_samples=None,n_test_samples=None,n_tests=1,
                approx=True,method="global",mmap=False,n_jobs=1):

        classifier.__init__(self,n_folds=n_folds,
                n_class_samples=n_class_samples,n_test_samples=n_test_samples,
                n_tests=n_tests,name='src_classifier')


        #a class that will do class dictionary learning
        #of the data
        self.class_dict_coder = class_dict_coder
        self.sparse_coder = sparse_coder
        #self.n_folds = n_folds
        #the number of training samples to use
        #per class (the rest are testing). some
        #papers use this method instead of cross validation (e.g the LC KSVD paper)
        #self.n_class_samples = n_class_samples
        #maximum number of test samples
        #self.n_test_samples = n_test_samples
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.approx = approx
        #if method='global' then we apply the global SRC classifier
        #if method='local' then we apply the local SRC classifier
        self.method = method
        self.mmap = mmap
        self.class_dict_coder.mmap = self.mmap
        self.n_jobs = n_jobs

        self.D_common = None



    def train(self,X_train,y_train):
        '''train the classifier'''
        n_classes = len(set(y_train))
        if self.n_nonzero_coefs is not None:
            self.sparse_coder = sparse_encoder(algorithm='ormp',params={'n_nonzero_coefs':self.n_nonzero_coefs},
                                                mmap=self.mmap,n_jobs=1)
        elif self.tol is not None:
            self.sparse_coder = sparse_encoder(algorithm='ormp',params={'tol':self.tol},
                                                mmap=self.mmap,n_jobs=1)

        #training
        D = self.class_dict_coder(X_train,y_train)
        self.n_class_atoms = self.class_dict_coder.n_class_atoms

        from lyssa.dict_learn.experimental import find_bad_atoms
        atom_scores = find_bad_atoms(X_train,D,y_train,sparse_coder=self.sparse_coder,
                            n_class_atoms=self.n_class_atoms,threshold=None)

        self.D_common = D[:,atom_scores[:500]]
        Z_common = self.sparse_coder(X_train,self.D_common)
        R = X_train - np.dot(self.D_common,Z_common)

        sparse_coder = sparse_encoder(algorithm='ormp',params={'n_nonzero_coefs':3},
                                            mmap=self.mmap,n_jobs=1)

        from lyssa.dict_learn.ksvd import class_ksvd_coder
        ckc = class_ksvd_coder(atom_ratio=0.2,sparse_coder=sparse_coder,
                        max_iter=3,approx=False,verbose=False,n_jobs=1)
        self.D = ckc(R,y_train)
        self.n_class_atoms = ckc.n_class_atoms
        #import pdb
        #pdb.set_trace()
        #testing



    def predict(self,X_test):
        '''test the classifier'''

        Z_common = self.sparse_coder(X_test,self.D_common)
        R = X_test - np.dot(self.D_common,Z_common)

        sparse_coder = sparse_encoder(algorithm='ormp',params={'n_nonzero_coefs':3},
                                            mmap=self.mmap,n_jobs=1)

        y_pred = src_predict(R,self.D,self.n_class_atoms,sparse_coder,
                            method=self.method,n_jobs=self.n_jobs)

        return y_pred
"""
