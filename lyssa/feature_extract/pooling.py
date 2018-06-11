import numpy as np


class sc_max_pooling:
    """applies max pooling on the absolute values of the sparse codes in Z"""
    def __call__(self, Z):
        return np.max(np.abs(Z), axis=1)


class max_pooling:
    """applies max pooling on the the sparse codes in Z"""
    def __call__(self, Z):
        return np.max(Z, axis=1)


class sum_pooling:
    """applies sum pooling on the sparse codes in Z"""
    def __call__(self, Z):
        return np.sum(Z, axis=1)


class average_pooling:
    """applies average pooling on the sparse codes in Z"""
    def __call__(self, Z):
        n_descriptors = Z.shape[1]
        return np.sum(Z, axis=1) / float(n_descriptors)
