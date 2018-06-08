from __future__ import division
import numpy as np
from lyssa.utils.math import fast_dot, norm_cols, norm, normalize, frobenius_squared
from lyssa.utils import set_openblas_threads


def average_mutual_coherence(D):
    n_atoms = D.shape[1]
    G = np.abs(np.dot(D.T, D))
    np.fill_diagonal(G, 0)
    return np.sum(G) / float(n_atoms * (n_atoms - 1))


def approx_error(D, Z, X, n_jobs=1):
    """computes the approximation error ||X-DZ||_{F}^{2} """
    if n_jobs > 1:
        set_openblas_threads(n_jobs)
    error = frobenius_squared(X - fast_dot(D, Z))
    return error


def approx_error_proc(X, Z, D):
    error = frobenius_squared(X - fast_dot(D, Z))
    return error


"""
def select_datapoints(X):
    n_samples = X.shape[1]
    idxs = [i for i in xrange(n_samples) if np.sum(X[:,i]**2) > 1e-6]
    return idxs
"""


def init_dictionary(X, n_atoms, method='data', return_unused_data=False, normalize=True):
    """ create the initial dictionary with n_atoms method: can be {data,svd,random}"""
    if method == "svd":
        from numpy.linalg import svd
        D, S, Z = svd(X, full_matrices=False)
        Z = S[:, np.newaxis] * Z
        r = len(Z)
        if n_atoms <= r:
            D = D[:, :n_atoms]
            Z = Z[:n_atoms, :]
        else:
            D = np.c_[D, np.zeros((len(D), n_atoms - r))]
            Z = np.r_[Z, np.zeros((n_atoms - r, Z.shape[1]))]

    elif method == "data":
        # select atoms randomly from the dataset
        # make sure they have non-zero norms(to avoid singular matrices)
        from numpy.linalg import norm
        from time import time
        n_samples = X.shape[1]
        idxs = [i for i in xrange(n_samples) if np.sum(X[:, i] ** 2) > 1e-6]

        if len(idxs) < n_atoms:
            print "not enough datapoints to initialize the dictionary"
            raise ValueError("not enough datapoints to initialize the dictionary")

        subset = np.random.choice(len(idxs), size=n_atoms, replace=False)
        subset_idxs = np.array(idxs).astype(int)[subset]

        D = X[:, subset_idxs]
        if normalize:
            D = norm_cols(D)
        if return_unused_data:
            s = set(subset_idxs)
            unused_data = [x for x in idxs if x not in s]
            return D, unused_data

    elif method == "random":
        D = np.random.randn(n_features, n_atoms)
        D = norm_cols(D)
    return D


def get_class_atoms(cl, n_class_atoms):
    # return the indices of the sub-dictionary of the cth class
    start = np.sum(n_class_atoms[:cl])
    end = start + n_class_atoms[cl]
    return range(int(start), int(end))


def force_mi(D, X, Z, unused_data, eta, max_tries=100):
    # force mutual incoherence within a dictionary
    n_atoms = D.shape[1]
    G = np.abs(np.dot(D.T, D))
    np.fill_diagonal(G, 0)

    for atom_idx1 in range(n_atoms):

        atom_idx2 = np.argmax(G[atom_idx1, :])
        # the maximum coherence
        mcoh = G[atom_idx1, atom_idx2]
        if mcoh < eta:
            print "less than the eta"
            continue
        # choose one of the two to replace
        # should we choose the one least used?
        if norm(Z[atom_idx1, :]) > norm(Z[atom_idx2, :]):
            c_atom = atom_idx1
        else:
            c_atom = atom_idx2

        # new_atom = None
        cnt = 0
        available_data = unused_data[:]
        min_idx = None
        min_coh = mcoh

        while mcoh > eta:
            # replace the coherent atom
            if cnt > max_tries:
                break
            # no datapoint available to be used as atom
            if len(available_data) == 0:
                return D
            _idx = np.random.choice(available_data, size=1)
            if len(_idx) == 0:
                return D, unused_data
            idx = _idx[0]
            new_atom = X[:, idx]
            new_atom = normalize(new_atom)
            available_data.remove(idx)
            g = np.abs(np.dot(D.T, new_atom))
            mcoh = np.max(g)
            if mcoh < min_coh or min_coh is None:
                min_coh = mcoh
                min_idx = idx
            cnt += 1

        D[:, c_atom] = X[:, min_idx]
        D[:, c_atom] = normalize(D[:, c_atom])
        unused_data.remove(min_idx)

    return D, unused_data


def find_coherent_atoms(Dc, Dj, thresh=None, kappa=None):
    # find coherent atoms between two dictionaries
    # i.e pairs of atoms that their coherence is greater
    # then the threshold thresh
    G = np.abs(np.dot(Dc.T, Dj))
    if kappa is not None:
        # idx contains the indices of the top kappa atoms
        # _idx = G.argsort(axis=None)[-alpha:][::-1]
        # c_atoms = [(idx/X.shape[1],idx%X.shape[1]) for idx in _idx]
        c_atoms = None
    if thresh is not None:
        _id = np.where(G > thresh)
        if len(_id[0]) == 0:
            return []
        c_atoms = [(_id[0][i], _id[1][i]) for i in range(len(_id[0]))]

    # return the list of pairs of coherent atoms
    # each element is a tuple of the form (k,l)
    # which means that the kth atom of Dc is coherent with
    # the lth atom of Dj
    return c_atoms


def replace_coherent_atoms(X, y, D, n_class_atoms, thresh=None, kappa=None, unused_data=None):
    # replace the coherent atoms of the sub-dictionaries in D
    n_classes = len(n_class_atoms)
    Xc = []
    for c in range(n_classes):
        x_c = y == c
        # extract the datapoints for
        # this class
        Xc.append(X[:, x_c])

    dicts = []
    for c in range(n_classes):
        dicts.append(D[:, get_class_atoms(c, n_class_atoms)])
    for c in range(n_classes):
        Dc = dicts[c]
        for j in range(c + 1, n_classes):
            Dj = dicts[j]
            c_atom_pairs = find_coherent_atoms(Dc, Dj, thresh=thresh, kappa=kappa)

            if unused_data is None:
                # we remove the coherent atoms from both Dc and Dj
                for c_atoms in c_atom_pairs:
                    Dc = np.delete(Dc, c_atoms[0], 1)
                    Dj = np.delete(Dj, c_atoms[1], 1)
                    n_class_atoms[c] = n_class_atoms[c] - 1
                    n_class_atoms[j] = n_class_atoms[j] - 1
            else:

                # replace them with one datapoint that
                # hasn't been used
                for c_atoms in c_atom_pairs:
                    # there are datapoints available
                    # to be used as atoms
                    if len(unused_data[c]) != 0:
                        _idx = np.random.choice(unused_data[c], size=1)
                        idx = _idx[0]
                        Dc[:, c_atoms[0]] = Xc[c][:, idx]
                        Dc[:, c_atoms[0]] = normalize(Dc[:, c_atoms[0]])
                        unused_data[c].remove(idx)

                    if len(unused_data[j]) != 0:
                        _idx = np.random.choice(unused_data[j], size=1)
                        idx = _idx[0]
                        Dj[:, c_atoms[1]] = Xc[j][:, idx]
                        Dj[:, c_atoms[1]] = normalize(Dj[:, c_atoms[1]])
                        unused_data[j].remove(idx)

                    dicts[c], dicts[j] = Dc, Dj

    # merge the sub-dictionaries
    D = dicts[0]
    for c in range(1, n_classes):
        D = np.hstack((D, dicts[c]))

    return D, n_class_atoms


"""
def find_shared_atoms(X,D,y,sparse_coder=None,n_class_atoms=None,threshold=None):

    #encode the datapoint over the joint dictionary
    Z = sparse_coder(X,D)

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

            #sometimes we get Index errors
            #
            if atom_idx == Z.shape[0]:
                break
            #print start_idx
            #find the indices of the datapoints that
            #use this atom
            data_idx = Z[atom_idx,:].nonzero()[0]
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
            atom_scores[atom_idx] = atom_score
            #idea2:
            #find all the datapoints that use this atom but belong to
            #a different class
            #diff_idx = set(data_idx).difference(set(class_idx))

            #find the coeffs in absolute value
            #np.abs(Z[atom_idx,diff_idx])
    #return the sorted indices of the atoms
    #descending order
    #atom_scores = np.fliplr([np.argsort(atom_scores)])[0]
    #ascending order
    atom_scores = np.argsort(atom_scores)
    return atom_scores
"""
