import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

# TODO: Create __all__ parameters to avoid suggesting imported stuff(?)


class BaseDetector(BaseEstimator, TransformerMixin):
    """
    To handle common Detector functionality like checking data, ensuring
    parallelization, reproducibility etc.
    """
    def __init__(self, n_jobs=1, random_state=None):
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _check_everything(self, X, y):
        """Checks hyper-parameters & data"""

        # Check/set random state
        rns = check_random_state(self.random_state)
        for k, v in self.get_params().items():
            if isinstance(v, BaseEstimator) and 'random_state' in v.get_params():
                v.set_params(random_state=rns.randint(10**8))

        self.classes_ = np.unique(y)

        return self._validate_data(X, y)

    def detect(self, X, y):
        raise NotImplementedError("")

    def fit_transform(self, X, y=None, **fit_params):
        return X, y, self.detect(X, y)

    def transform(self, X):
        return X
