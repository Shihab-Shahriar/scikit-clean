import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors._base import _get_weights
from sklearn.utils.extmath import weighted_mode


# TODO: support all sklearn Random Forest parameters
class RobustForest(BaseEstimator, ClassifierMixin):
    """
    Uses a random forest to to compute pairwise similarity/distance, and then \
    a simple K Nearest Neighbor that works on that similarity matrix. For
    a pair of samples, the similarity value is proportional to how frequently \
    they belong to the same leaf. See :cite:`forestkdn17` for details.

    Parameters
    ------------
    method : string, default='simple'
        There are two different ways to compute similarity matrix. In 'simple'
        method, the similarity value is simply the percentage of times two \
        samples belong to same leaf. 'weighted' method also takes the size of \
        those leaves into account- it exactly matches above paper's algorithm, \
        but it is computationally slow.

    K : int, default=5
        No of nearest neighbors to consider for final classification

    n_estimators : int, default=101
        No of trees in Random Forest.

    max_leaf_nodes : int, default=128
        Maximum no of leaves in each tree.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility

    """

    def __init__(self, method='simple', K=5, n_estimators=100, max_leaf_nodes=128,
                 random_state=None, n_jobs=None):
        self.method = method
        self.K = K
        self.n_estimators = n_estimators
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.forest_ = RandomForestClassifier(
            n_estimators=self.n_estimators, max_leaf_nodes=self.max_leaf_nodes,
            n_jobs=self.n_jobs, random_state=self.random_state
        ).fit(X, y)
        self.data_ = (X, y)
        return self

    def _pairwise_distance_weighted(self, train_X, test_X):
        out_shape = (test_X.shape[0], train_X.shape[0])
        mat = np.zeros(out_shape, dtype='float32')
        temp = np.zeros(out_shape, dtype='float32')
        to_add = np.zeros(out_shape, dtype='float32')
        ones = np.ones(out_shape, dtype='float32')
        for tree in self.forest_.estimators_:
            train_leaves = tree.apply(train_X)
            test_leaves = tree.apply(test_X)
            match = test_leaves.reshape(-1, 1) == train_leaves.reshape(1, -1)  # Samples w/ same leaf as mine:mates
            no_of_mates = match.sum(axis=1, dtype='float')  # No of My Leaf mates
            np.multiply(match, no_of_mates.reshape(-1, 1),
                        out=temp)  # assigning weight to each leaf-mate, proportional to no of mates
            to_add.fill(0)
            np.divide(ones, temp, out=to_add, where=temp != 0)  # Now making that inversely proportional
            assert np.allclose(to_add.sum(axis=1), 1)
            assert match.shape == (len(test_X), len(train_X)) == to_add.shape == temp.shape
            assert no_of_mates.shape == (len(test_X),)
            np.add(mat, to_add, out=mat)
        return 1 - mat / len(self.forest_.estimators_)

    def _pairwise_distance_simple(self, train_X, test_X):
        train_leaves = self.forest_.apply(train_X)  # (train_X,n_estimators)
        test_leaves = self.forest_.apply(test_X)  # (test_X,n_estimators)
        dist = cdist(test_leaves, train_leaves, metric='hamming')
        assert dist.shape == (len(test_X), len(train_X))
        return dist

    def pairwise_distance(self, train_X, test_X):
        if self.method == 'simple':
            return self._pairwise_distance_simple(train_X, test_X)
        elif self.method == 'weighted':
            return self._pairwise_distance_weighted(train_X, test_X)
        raise Exception("method not recognized")

    def predict(self, X):
        train_X, train_Y = self.data_
        dist = self.pairwise_distance(train_X, X)
        assert np.all(dist >= 0)
        idx = np.argsort(dist, axis=1)
        nn_idx = idx[:, :self.K]
        nn_dist = dist[np.arange(len(X))[:, None], nn_idx]
        nn_labels = train_Y[nn_idx]
        weights = _get_weights(nn_dist, 'distance')  # Weighted KNN
        a, _ = weighted_mode(nn_labels, weights, axis=1)
        return a.reshape(-1)
