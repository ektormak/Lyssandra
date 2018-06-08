from __future__ import division
from sklearn.utils import as_float_array
import numpy as np
from scipy.linalg import eigh
from lyssa.utils.math import normalize, norm_cols


class l2_normalizer():
    def __call__(self, Z):
        if Z.ndim == 1:
            return normalize(Z)
        elif Z.ndim == 2:
            # e.g a 2D patch in a feature_map
            shape = Z.shape
            return normalize(Z.flatten()).reshape(shape)


class ZCA():
    # each datapoint is a row of X

    def __init__(self, n_components=None, bias=.1, copy=False):
        self.n_components = n_components
        self.bias = bias
        self.copy = copy

    def fit(self, X, y=None):
        # X = array2d(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        # substracts the mean for each feature vector
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        eigs, eigv = eigh(np.dot(X.T, X) / n_samples + \
                          self.bias * np.identity(n_features))
        components = np.dot(eigv * np.sqrt(1.0 / eigs), eigv.T)
        self.components_ = components
        # Order the explained variance from greatest to least
        self.explained_variance_ = eigs[::-1]
        return self

    def transform(self, X):
        if self.mean_ is not None:
            X -= self.mean_
        X_transformed = np.dot(X, self.components_)
        return X_transformed


def local_contrast_normalization(X):
    """apply local constrast normalization to the datapoints in X"""
    X = X.T
    X = X.reshape((X.shape[0], -1))
    X -= X.mean(axis=1)[:, None]
    X_std = X.std(axis=1)
    # This trick is credited to N. Pinto
    min_divisor = (2 * X_std.min() + X_std.mean()) / 3
    X /= np.maximum(min_divisor, X_std).reshape(
        (X.shape[0], 1))
    return X.T


class preproc():
    def __init__(self, name):
        self.name = name

    def __call__(self, X):

        # assumes that each datapoint is a
        # column of the 2D matrix X

        if self.name == 'global_centering':
            # remove the mean of each feature vector
            X = X - X.mean(axis=1)[:, np.newaxis]
        elif self.name == 'global_standarization':
            # remove the mean and
            # divide by the variance of each feature vector
            X = X - X.mean(axis=1)[:, np.newaxis]
            X = X / X.std(axis=1)[:, np.newaxis]
        elif self.name == 'local_centering':
            # remove the mean of each datapoint
            X = X - X.mean(axis=0)[np.newaxis, :]
        elif self.name == 'contrast_normalization':
            # remove the mean of each datapoint
            # and divide by its norm
            X = X - X.mean(axis=0)[np.newaxis, :]
            X = norm_cols(X)
        elif self.name == 'normalization':
            return norm_cols(X)
        elif self.name == 'scaling':
            # scale each pixel in an image
            # to lie in[0,1]
            X = X / 255.
        elif self.name == 'whitening':
            X = ZCA().fit(X.T).transform(X.T).T

        return X
