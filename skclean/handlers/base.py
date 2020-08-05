import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state

from ..detectors.base import BaseDetector
from sklearn.utils.validation import _check_sample_weight


def _check_data_params(obj, X, y, conf_score):
    """Extracted out of BaseHandler for WeightedBag & Costing"""

    # Reproducibility
    rns = check_random_state(obj.random_state)
    for k, v in obj.get_params().items():
        if isinstance(v, BaseEstimator) and 'random_state' in v.get_params():
            v.set_params(random_state=rns.randint(10**8))

    # Parallelization
    if obj.classifier and 'n_jobs' in obj.classifier.get_params():
        obj.classifier.set_params(n_jobs=obj.n_jobs)
    if obj.detector and 'n_jobs' in obj.detector.get_params():
        obj.detector.set_params(n_jobs=obj.n_jobs)

    if conf_score is None and obj.detector is None:
        raise ValueError("Neither conf_score or Detector is supplied to Handler")

    if conf_score is None:  # outside Pipeline/ inside Iterative Handler
        conf_score = obj.detector.detect(X, y)

    X, y = obj._validate_data(X, y)
    obj.classes_ = np.unique(y)
    conf_score = _check_sample_weight(conf_score, X)
    return X, y, conf_score


# Non-iterative Handlers can be used both w/ pipeline and H(c=C(),d=D()) format
class BaseHandler(BaseEstimator):
    def __init__(self, classifier=None, detector=None, *, n_jobs=1, random_state=None):
        self.classifier = classifier
        self.detector = detector
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _check_everything(self, X, y, conf_score):
        """Check hyparams suppiled in __init__ & data"""
        return _check_data_params(self, X, y, conf_score)

    def fit(self, X, y, conf_score=None):
        raise NotImplementedError("Attempt to instantiate abstract class")

    # problem with ensemble handlers: i.e. many copies of obj.classifier
    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    @property
    def iterative(self):  # Does this Handler call Detector multiple times?
        return False

