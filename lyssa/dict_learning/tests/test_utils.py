import unittest

import numpy as np

from lyssa.dict_learning.utils import get_class_atoms, init_dictionary


class UtilsTest(unittest.TestCase):

    def test_get_class_atoms(self):
        n_class_atoms = [2, 3, 5]
        cl = 1
        self.assertEqual(get_class_atoms(cl, n_class_atoms=n_class_atoms), [2, 3, 4])

    def test_init_dictionary(self):
        X = np.array([
            [1, 2, 3, 4, 5],
            [0, 2, 1, 2, 1]
        ])
        n_datapoints = 5
        n_atoms = 3

        D = init_dictionary(X, n_atoms, method='data', return_unused_data=False, normalize=False)

        self.assertEqual(D.shape, (2, n_atoms))
        self.assertEqual(np.sum(np.array_equal(D[:, i], X[:, j])
                                for i in range(n_atoms) for j in range(n_datapoints)), n_atoms)
