import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors._base import _get_weights

from .base import BaseDetector


# TODO: Support other distance metrics
class KDN(BaseDetector):
    """
    For each sample, the percentage of it's nearest neighbors with same
    label serves as it's `conf_score`. Euclidean distance is used to
    find the nearest neighbors. See :cite:`ensih18,ih14` for details.


    Parameters
    --------------
    n_neighbors : int, default=5
        No of nearest neighbors to use to compute `conf_score`

    weight : string, default='uniform'
        weight function used in prediction. If 'uniform', all points
        in each neighborhood are weighted equally. If 'distance', weights
        points by the inverse of their distance.

    n_jobs : int, default=1
        No of parallel cpu cores to use
    """

    def __init__(self, n_neighbors=5, weight='uniform', n_jobs=1):
        super().__init__(n_jobs=n_jobs, random_state=None)
        self.n_neighbors = n_neighbors
        self.weight = weight

    def _get_kdn(self, knn, y):
        dist, kid = knn.kneighbors()  # (n_estimators,K) : ids & dist of nn's for every sample in X
        weights = _get_weights(dist, self.weight)
        if weights is None:
            weights = np.ones_like(kid)
        agreement = y[kid] == y.reshape(-1, 1)
        return np.average(agreement, axis=1, weights=weights)

    def detect(self, X, y):
        X, y = self._validate_data(X, y)  # , accept_sparse=True
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weight,
                                   n_jobs=self.n_jobs).fit(X, y)
        return self._get_kdn(knn, y)


class ForestKDN(KDN):
    """
    Like KDN, but a trained Random Forest is used to compute pairwise similarity.

    Specifically, for a pair of samples, their similarity is the percentage of
    times they belong to the same leaf. See :cite:`forestkdn17` for details.

    Parameters
    -------------------
    n_neighbors : int, default=5
        No of nearest neighbors to use to compute `conf_score`

    n_estimators : int, default=101
        No of trees in Random Forest.

    max_leaf_nodes : int, default=64
        Maximum no of leaves in each tree.

    weight : string, default='distance'
        weight function used in prediction. If 'distance', weights
        points by the inverse of their distance. If 'uniform', all points
        in each neighborhood are weighted equally.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility
    """

    def __init__(self, n_neighbors=5, n_estimators=100, max_leaf_nodes=64,
                 weight='distance', n_jobs=1, random_state=None):
        super().__init__(n_neighbors=n_neighbors, weight=weight, n_jobs=n_jobs)
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state

    def detect(self, X, y):
        X, y = self._check_everything(X, y)

        forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_leaf_nodes=self.max_leaf_nodes, n_jobs=self.n_jobs,
            random_state=self.random_state).fit(X, y)

        Xs = forest.apply(X)
        knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, metric='hamming', algorithm='brute',
            weights=self.weight, n_jobs=self.n_jobs).fit(Xs, y)

        return self._get_kdn(knn, y)


# TODO: rename this class (?)
class HybridKDN(KDN):
    def __init__(self, classifier, n_neighbors=5, weight='uniform', n_jobs=1):
        super().__init__(n_neighbors=n_neighbors, weight=weight, n_jobs=n_jobs)
        self.classifier = classifier

    def detect(self, X, y):
        X, y = self._validate_data(X, y)

        try:  # classifier may already be trained
            yp = self.classifier.predict(X)
        except NotFittedError:
            yp = self.classifier.fit(X, y).predict(X)

        knn = KNeighborsClassifier().fit(X, y)
        _, kid = knn.kneighbors()
        agr = yp[kid] == y[kid]
        return agr.sum(axis=1) / knn.n_neighbors


class RkDN(KDN):
    __doc__ = KDN.__doc__

    def detect(self, X, y):
        X, y = self._validate_data(X, y)
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weight,
                                   n_jobs=self.n_jobs).fit(X, y)
        _, kid = knn.kneighbors()

        N = len(X)
        M = np.zeros((N, N), dtype='bool')

        cols = np.zeros_like(kid) + np.arange(0, N).reshape(-1, 1)
        M[kid.reshape(-1), cols.reshape(-1)] = 1

        label_agr = y.reshape(1, -1) == y.reshape(-1, 1)
        agr = M & label_agr

        m = M.sum(axis=1).astype('float')

        # Outliers who doesn't belong to anybody's NN list have conf_score=0
        return np.divide(agr.sum(axis=1), m, out=np.zeros_like(m), where=(m != 0))
