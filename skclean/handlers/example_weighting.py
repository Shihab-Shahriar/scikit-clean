import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

from skclean.handlers.base import BaseHandler, _check_data_params


class SampleWeight(BaseHandler, ClassifierMixin):
    """
    Simply passes `conf_score` (computed with `detector`) as sample weight
    to underlying classifier.

    Parameters
    ------------
    classifier: object
        A classifier instance supporting sklearn API. Must support `sample_weight`
        in `fit()` method.

    detector : `BaseDetector` or None, default=None
        To compute `conf_score`. Set it to `None` only if `conf_score` is \
        expected in `fit()` (e.g. when used inside a Pipeline with a \
        `BaseDetector` preceding it). Otherwise a Detector must be supplied \
        during instantiation.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility
    """

    def __init__(self, classifier, detector=None, *, n_jobs=1, random_state=None):
        super().__init__(classifier, detector, n_jobs=n_jobs,
                         random_state=random_state)

    def fit(self, X, y, conf_score=None):
        X, y, conf_score = self._check_everything(X, y, conf_score)
        self.classifier.fit(X, y, sample_weight=conf_score)
        return self


class Relabeling:
    """
    Flip labels when confident about some other samples. Will require access to
    decision_function (predict_proba i.e. N*L shape).

    Raise error early if detector doesn't support it. Also, show available detectors
    that do support it by providing sample code to run to get that list.

    Find another Relabeling algo, and Put them in a 3rd file?
    """


class _WBBase(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, replacement, sampling_ratio, sample_weight=None):
        super().__init__()
        self.estimator = estimator
        self.replacement = replacement
        self.sampling_ratio = sampling_ratio
        self.sample_weight = sample_weight

    def fit(self, X, y):
        rng = check_random_state(self.estimator.random_state)
        to_sample = int(self.sampling_ratio * len(y))
        target_idx = rng.choice(len(y), size=to_sample, replace=self.replacement, p=self.sample_weight)
        X, y = X[target_idx], y[target_idx]
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class WeightedBagging(BaggingClassifier):
    """
    Similar to regular bagging- except cleaner samples will be chosen
    more often during bagging. That is, a sample's probability of
    getting selected in bootstrapping process is directly proportional to
    it's `conf_score`. See :cite:`ensih18` for details.

    Parameters
    ------------------
    classifier: object
        A classifier instance supporting sklearn API. Same as `base_estimator`
        of scikit-learn's BaggingClassifier.

    detector : `BaseDetector` or None, default=None
        To compute `conf_score`. Set it to `None` only if `conf_score` is \
        expected in `fit()` (e.g. when used inside a Pipeline with a \
        `BaseDetector` preceding it). Otherwise a Detector must be supplied \
        during instantiation.

    n_estimators : int, default=10
        The number of base classifiers in the ensemble.

    replacement : bool, default=True
        Whether to sample instances with/without replacement at each base classifier

    sampling_ratio : float, 0.0 to 1.0, default=1.0
        No of samples drawn at each tree equals: len(X) * sampling_ratio

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility

    verbose : int, default=0
        Controls the verbosity when fitting and predicting

    """
    def __init__(self,
                 classifier=None,
                 detector=None,
                 n_estimators=100,
                 replacement=True,
                 sampling_ratio=1.0,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        BaggingClassifier.__init__(
            self,
            base_estimator=classifier,
            warm_start=False,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
            bootstrap=False,
            bootstrap_features=False,
            verbose=verbose)
        self.classifier = classifier
        self.detector = detector
        self.replacement = replacement
        self.sampling_ratio = sampling_ratio

    @property
    def classifier(self):
        return self.base_estimator

    @classifier.setter
    def classifier(self, clf):
        self.base_estimator = clf


    def _validate_estimator(self, default=DecisionTreeClassifier()):
        super()._validate_estimator()
        self.base_estimator_ = _WBBase(self.base_estimator_, self.replacement, self.sampling_ratio, self.conf_score_)

    def fit(self, X, y, conf_score=None, **kwargs):
        X, y, conf_score = _check_data_params(self, X, y, conf_score)
        conf_score += 1 / (len(y))
        self.conf_score_ = conf_score/conf_score.sum()  # Sum to one
        return super().fit(X, y)

    @property
    def iterative(self):  # Does this Handler call Detector multiple times?
        return False


class _RSBase(ClassifierMixin, BaseEstimator):  # Rejection sampling Base
    def __init__(self, estimator, sample_weight=None):
        super().__init__()
        self.estimator = estimator
        self.sample_weight = sample_weight

    def fit(self, X, y):
        rng = check_random_state(self.estimator.random_state)
        r = rng.uniform(self.sample_weight.min(), self.sample_weight.max(), size=y.shape)
        target_idx = r <= self.sample_weight

        if len(np.unique(y[target_idx])) != len(np.unique(y)):
            warnings.warn("One or more classes are not present after resampling")

        X, y = X[target_idx], y[target_idx]
        self.estimator = self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


class Costing(BaggingClassifier):
    """
    Implements *costing*, a method combining cost-proportionate rejection
    sampling and ensemble aggregation. At each base classifier, samples
    are selected for training with probability equal to `conf_score`.
    See :cite:`costing03` for details.


    Parameters
    ------------------
    classifier: object
        A classifier instance supporting sklearn API. Same as `base_estimator`
        of scikit-learn's BaggingClassifier.

    detector : `BaseDetector` or None, default=None
        To compute `conf_score`. Set it to `None` only if `conf_score` is \
        expected in `fit()` (e.g. when used inside a Pipeline with a \
        `BaseDetector` preceding it). Otherwise a Detector must be supplied \
        during instantiation.

    n_estimators : int, default=10
        The number of base classifiers in the ensemble.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility

    verbose : int, default=0
        Controls the verbosity when fitting and predicting
    """
    def __init__(self,
                 classifier=None,
                 detector=None,
                 n_estimators=100,
                 n_jobs=1,
                 random_state=None,
                 verbose=0):
        BaggingClassifier.__init__(
            self,
            base_estimator=classifier,
            n_estimators=n_estimators,
            warm_start=False,
            n_jobs=n_jobs,
            random_state=random_state,
            bootstrap=False,
            bootstrap_features=False,
            verbose=verbose)
        self.classifier = classifier
        self.detector = detector

    @property
    def classifier(self):
        return self.base_estimator

    @classifier.setter
    def classifier(self, clf):
        self.base_estimator = clf

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        super()._validate_estimator()
        self.base_estimator_ = _RSBase(self.base_estimator_, self.conf_score_)

    # Duplicate fit
    def fit(self, X, y, conf_score=None, **kwargs):
        X, y, conf_score = _check_data_params(self, X, y, conf_score)
        conf_score += 1 / (len(y))
        self.conf_score_ = conf_score/conf_score.sum()  # Sum to one
        return super().fit(X, y)

    @property
    def iterative(self):  # Does this Handler call Detector multiple times?
        return False
