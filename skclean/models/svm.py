"""Unhinged loss"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import *


class Centroid(BaseEstimator, ClassifierMixin):
    """It seems equal/better than sklearn's NearestCentroid, but on spambase
    it performs really bad - nope, I just forgot to scale. Need to support other labels"""
    def __init__(self, kernel='rbf'):
        self.kernel = kernel

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.data_ = {}
        for c in np.unique(y):
            self.data_[c] = X[y == c]
        return self

    def predict(self, X):
        dist = np.zeros((len(X), len(self.data_)))
        for c in self.data_:
            dist[:, c] = pairwise_kernels(X, self.data_[c], metric=self.kernel).mean(axis=1)
        return np.argmax(dist, axis=1)

