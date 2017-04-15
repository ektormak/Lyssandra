import numpy as np

"""
common pooling operators

sc_max_pooling:

applies max pooling on the absolute values
of the sparse codes in Z(each stored in a column)

max_pooling:

applies max pooling on the
the sparse codes in Z(each stored in a column)

sum_pooling:

applies sum pooling on the
sparse codes in Z(each stored in a column)

average_pooling:

applies average pooling on the
sparse codes in Z(each stored in a column)

"""


class sc_max_pooling:
    def __call__(self, Z):
        return np.max(np.abs(Z), axis=1)


class max_pooling:
    def __call__(self, Z):
        return np.max(Z, axis=1)


class sum_pooling:
    def __call__(self, Z):
        return np.sum(Z, axis=1)


class average_pooling:
    def __call__(self, Z):
        n_descriptors = Z.shape[1]
        return np.sum(Z, axis=1) / float(n_descriptors)
