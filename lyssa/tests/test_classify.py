import unittest

import numpy as np

from lyssa.classify import class_accuracy, avg_class_accuracy


class ClassifyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(ClassifyTest, cls).setUpClass()
        cls.y = np.array([0, 0, 2, 1, 0, 2, 1, 2, 0, 1])
        cls.y_pred = np.array([1, 0, 1, 1, 0, 2, 2, 2, 1, 0])

    def test_class_accuracy(self):
        self.assertEqual(class_accuracy(self.y, self.y_pred), 0.5)

    def test_avg_class_accuracy(self):
        # class 0 has 0.5 accuracy
        # class 1 has 0.33 accuracy
        # class 2 has 0.66 accuracy
        self.assertEqual(round(avg_class_accuracy(self.y_pred, self.y), 2), 0.5)