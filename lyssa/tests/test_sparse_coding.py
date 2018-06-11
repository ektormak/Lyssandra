import unittest

import numpy as np

from lyssa.sparse_coding import sparse_encoder


class SparseCodingTest(unittest.TestCase):

    def test_invalid_encoder(self):
        n_features = 10
        n_atoms = 4
        n_nonzero_coefs = 4
        n_datapoints = 100
        X = np.random.rand(n_features, n_datapoints)
        D = np.random.rand(n_features, n_atoms)
        se = sparse_encoder(algorithm='se', params={'n_nonzero_coefs': n_nonzero_coefs})
        with self.assertRaises(Exception):
            Z = se.encode(X, D)
