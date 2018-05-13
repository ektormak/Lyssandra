import numpy as np
from utils.dataset import split_dataset
import abc
from sklearn import svm
from sklearn.grid_search import ParameterGrid
from sklearn.cross_validation import StratifiedKFold


def class_accuracy(y_pred, y_test):
    # the classification accuracy
    n_correct = np.sum(y_test == y_pred)
    return n_correct / float(y_test.size)


def avg_class_accuracy(y_pred, y_test):
    """
    the classification accuracy averaged over the classes
    """
    n_classes = len(set(y_test))
    class_accs = []

    for c in range(n_classes):
        n_correct = np.sum(y_test[y_test == c] == y_pred[y_test == c])
        n_class_samples = y_test[y_test == c].size
        class_accs.append(n_correct / float(n_class_samples))

    return np.mean(class_accs)


def average_precision(tp, fp, n_pos):
    """
    computes the average precision given the
    true and false positive predictions.

    tp and fp should be vectors
    with length equal to the number of samples
    if tp[i] = 1 then the ith sample
    is a true positive
    if fp[i] = 1 then the ith sample
    is a false positive
    e.g:
    y_pred = np.array([1,1,1,1,1,1,0,0,0,1]).astype(int)
    y_test = np.array([1,0,1,0,1,1,1,1,1,0]).astype(int)
    """

    n_samples = len(tp)

    """
    #case the user provides
    #y_test and y_pred instead
    tp = np.zeros(n_samples)
    fp = np.zeros(n_samples)

    n_pos = np.sum(y_test==1)

    for i in range(n_samples):
        if y_pred[i] == 1 and y_pred[i] == y_test[i]:
            tp[i] = 1
        if y_test[i] == 0 and y_pred[i] == 1:
            fp[i] = 1
    """
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    rec = tp / float(n_pos)
    prec = tp / (fp + tp)

    mrec = np.zeros(n_samples + 2)
    mrec[0] = 0
    mrec[-1] = 1
    mrec[1:-1] = rec

    mpre = np.zeros(n_samples + 2)
    mpre[0] = 0
    mpre[-1] = 0
    mpre[1:-1] = prec
    for i in range(n_samples - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    idx = (mrec[1:] != mrec[:-1]).nonzero()[0] + 1

    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx])
    return ap


class classifier():
    """
    an abstract class that models a classifier
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, param_grid=None, n_folds=None,
                 n_class_samples=None, n_test_samples=None, n_tests=1, name="classifier"):
        self.name = name
        self.param_grid = param_grid
        self.best_param_set = None
        self.n_folds = n_folds
        # the number of validation or test samples per class
        self.n_test_samples = n_test_samples
        # the number of training samples per class
        self.n_class_samples = n_class_samples
        self.n_tests = n_tests

    def fit(self, X, y):
        self.__call__(X, y)

    def __call__(self, X, y):
        """
        given a dataset X,y we split it, in order to do cross validation,
        according to the procedure explained below:
        if n_folds is not None, then we do cross validation
        based on stratified folds
        if n_class_samples is not None, then we do cross validation
        using only <n_class_samples> training samples per class
        if n_test_samples is not None, then we do cross validation
        using only <n_test_samples> cross validaition samples per class
        assumes that each datapoint is in a column of X
        """
        n_classes = len(set(y))
        if self.n_folds is not None:
            # generate the folds
            self.folds = StratifiedKFold(y, n_folds=self.n_folds,
                                         shuffle=False, random_state=None)

        elif self.n_class_samples is not None:

            self.folds = []
            for i in range(self.n_tests):

                if type(self.n_class_samples) is not list:
                    self.n_class_samples = (np.ones(n_classes) * self.n_class_samples).astype(int)
                if self.n_test_samples is not None:
                    self.n_test_samples = (np.ones(n_classes) * self.n_test_samples).astype(int)

                data_idx = split_dataset(self.n_class_samples, self.n_test_samples, y)
                train_idx = data_idx[0]
                test_idx = data_idx[1]
                self.folds.append((train_idx, test_idx))

        self.cross_validate(X, y)

    def cross_validate(self, X, y):

        print "fitting {} to the training set".format(self.name)
        if self.param_grid is not None:
            param_sets = list(ParameterGrid(self.param_grid))
            n_param_sets = len(param_sets)
            param_scores = []
            for j, param_set in enumerate(param_sets):

                print "--------------"
                print "training the classifier..."
                print "parameter set:"
                for k, v in param_set.iteritems():
                    print "{}:{}".format(k, v)

                param_score = self.evaluate(X, y, param_set=param_set)
                param_scores.append(param_score)
                p = np.argmax(np.array(param_scores))
                self.best_param_set = param_sets[p]
                print "best parameter set", self.best_param_set
                print "best score:", param_scores[p]
        else:
            score = self.evaluate(X, y)

    def evaluate(self, X, y, param_set=None):
        """
        evaluate the performance of the classifier
        trained with the parameters in <param_set>
        """
        cv_scores = []
        # avg_class_accs = []
        for train_index, test_index in self.folds:
            X_train, X_test = X[:, train_index], X[:, test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.train(X_train, y_train, param_set=param_set)

            y_pred = self.predict(X_test)
            y_pred = np.array(y_pred)
            class_acc = class_accuracy(y_pred, y_test)
            # avg_class_acc  = avg_class_accuracy(y_pred,y_test)
            cv_scores.append(class_acc)
            # avg_class_accs.append(avg_class_acc)
            print "average class accuracy:", avg_class_accuracy(y_pred, y_test)

        avg_cv_score = np.mean(cv_scores)
        print "accuracy:", avg_cv_score
        return avg_cv_score

    @abc.abstractmethod
    def train(self, X_train, y_train, param_set=None):
        '''train the classifier'''
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X_test):
        '''test the classifier'''
        raise NotImplementedError


class linear_svm(classifier):
    """
    a wrapper to scikit's Linear SVM.
    """

    def __init__(self, param_grid=None, n_folds=None,
                 n_class_samples=None, n_test_samples=None, n_tests=1, name="linear svm classifier"):
        classifier.__init__(self, param_grid=param_grid, n_folds=n_folds,
                            n_class_samples=n_class_samples, n_test_samples=n_test_samples, n_tests=n_tests, name=name)

        if param_grid is None:
            self.param_grid = [{'C': [1e-10, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 1, 10]}]
        self.clf = svm.LinearSVC()
        self.clf.penalty = 'l2'
        self.clf.dual = False
        self.clf.multi_class = 'ovr'

    def train(self, X_train, y_train, param_set=None):
        '''train the classifier'''
        self.clf.set_params(**param_set)
        self.clf.fit(X_train.T, y_train)

    def predict(self, X_test):
        '''test the classifier'''
        y_pred = self.clf.predict(X_test.T)
        return y_pred
