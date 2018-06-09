import unittest

import numpy as np

from lyssa.dict_learning.gradient_descent import dictionary_learner
from lyssa.sparse_coding import sparse_encoder


class DictionaryLearningTest(unittest.TestCase):

    def test_grad_desc(self):
        n_features = 10
        n_atoms = 4
        n_nonzero_coefs = 4
        n_datapoints = 100
        X = np.random.rand(n_features, n_datapoints)
        se = sparse_encoder(algorithm='bomp', params={'n_nonzero_coefs': n_nonzero_coefs})
        dl = dictionary_learner(n_atoms=n_atoms, sparse_coder=se, eta=0.1, batch_size=None)
        Z = dl(X)
        self.assertEqual(Z.shape, (n_atoms, n_datapoints))
        self.assertEqual(dl.D.shape, (n_features, n_atoms))
