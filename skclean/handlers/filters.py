import warnings
import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle, check_random_state

from .base import BaseHandler


class Filter(BaseHandler, ClassifierMixin):
    """
    Removes from dataset samples most likely to be noisy. Samples-to-be-removed
    can be selected in two ways: either a specified percentage of samples
    with lowest `conf_score`, or samples with lower `conf_score` than a
    specified threshold.

    Parameters
    ----------------
    classifier: object
        A classifier instance supporting sklearn API.

    detector : `BaseDetector` or None, default=None
        To compute `conf_score`. Set it to `None` only if `conf_score` is \
        expected in `fit()` (e.g. when used inside a Pipeline with a \
        `BaseDetector` preceding it). Otherwise a Detector must be supplied \
        during instantiation.

    threshold: float, default=.5
        Samples with higher conf_score will be kept, rest will be filtered out. A \
        value of .5 implies majority voting, whereas .99 (i.e. a value closer to, \
        but less than 1.0) implies onsensus voting.

    frac_to_filter: float, default=None
        Percentages of samples to filter out. Exactly one of either threshold or \
        frac_to_filter must be set.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility

    """

    def __init__(self, classifier, detector=None, threshold: float = .5,
                 frac_to_filter: float = None,
                 n_jobs=1, random_state=None):
        super().__init__(classifier, detector, n_jobs=n_jobs, random_state=random_state)
        self.threshold = threshold
        self.frac_to_filter = frac_to_filter

    def fit(self, X, y, conf_score=None):
        X, y, conf_score = self._check_everything(X, y, conf_score)

        if not self.threshold and not self.frac_to_filter:
            raise ValueError("At least one of threshold or frac_to_filter must "
                             "be supplied")

        if self.threshold is not None and self.frac_to_filter is not None:
            raise ValueError("Both threshold and frac_to_filter can not be supplied "
                             "together, choose one.")

        if self.frac_to_filter is None:
            clean_idx = conf_score > self.threshold
        else:
            to_take = int(len(conf_score) * (1 - self.frac_to_filter))
            clean_idx = np.argsort(-conf_score)[:to_take]

        self.classifier.fit(X[clean_idx], y[clean_idx])
        return self


# TODO: Support RandomState obj everywhere
# TODO: Change all "See X for details" to details/usage
class FilterCV(BaseHandler, ClassifierMixin):
    """
    For quickly finding best cutoff point for Filter i.e. `threshold` or \
    `fraction_to_filter`. This avoids recomputing `conf_score` for each \
    hyper-parameter value as opposed to say GridSearchCV. See \
    :cite:`twostage18` for details/usage.

    Parameters
    -------------------
    classifier: object
        A classifier instance supporting sklearn API.

    detector : `BaseDetector` or None, default=None
        To compute `conf_score`. Set it to `None` only if `conf_score` is \
        expected in `fit()` (e.g. when used inside a Pipeline with a \
        `BaseDetector` preceding it). Otherwise a Detector must be supplied \
        during instantiation.

    thresholds : list, default=None
        A list of thresholds to choose the best one from

    fracs_to_filter : list, default=None
        A list of percentages to choose the best one from

    cv : int, cross-validation generator or an iterable, default=None
        If None, uses 5-fold stratified k-fold
        if int, no of folds to use in stratified k-fold

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility
    """

    def __init__(self, classifier, detector=None, thresholds=None,
                 fracs_to_filter=None, cv=5,
                 n_jobs=1, random_state=None):
        super().__init__(classifier, detector, n_jobs=n_jobs, random_state=random_state)
        self.thresholds = thresholds
        self.fracs_to_filter = fracs_to_filter
        self.cv = StratifiedKFold(n_splits=cv) if isinstance(cv, int) else cv

    def _get_clean_idx(self, point, conf_score):
        if self.thresholds is not None:
            return np.argwhere(conf_score > point).reshape(-1)
        to_take = int(len(conf_score) * (1 - point))
        return np.argsort(-conf_score)[:to_take]

    def _find_cutoff(self, X, y, conf_score):
        """Find the best cutoff point (either threshold or frac_to_filter)
        using cross_validation"""

        self.cv.random_state = check_random_state(self.random_state).randint(10 ** 8)

        cutoff_points = self.thresholds or self.fracs_to_filter
        best_acc, best_cutoff = 0.0, cutoff_points[0]
        for point in cutoff_points:
            clean_idx = self._get_clean_idx(point, conf_score)
            accs = []
            for tr_idx, test_idx in self.cv.split(X, y):
                train_idx = set(tr_idx).intersection(clean_idx)
                train_idx = np.array(list(train_idx))
                if len(train_idx) == 0:
                    warnings.warn("All training instances of identified as noisy, skipping this fold")
                    continue

                clf = clone(self.classifier).fit(X[train_idx], y[train_idx])
                acc = clf.score(X[test_idx], y[test_idx])
                accs.append(acc)

            avg_acc = sum(accs) / len(accs) if len(accs) > 0 else 0.0
            print(point, avg_acc)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_cutoff = point
        return best_cutoff

    def fit(self, X, y, conf_score=None):
        X, y, conf_score = self._check_everything(X, y, conf_score)

        cutoff = self._find_cutoff(X, y, conf_score)
        clean_idx = self._get_clean_idx(cutoff, conf_score)

        self.classifier.fit(X[clean_idx], y[clean_idx])
        return self


# TODO: Support frac_to_filter, maybe using Filter? - nah, w/o FIlter
class CLNI(BaseHandler, ClassifierMixin):
    """
    Iteratively detects and filters out mislabelled samples unless
    a stopping criterion is met. See :cite:`clni11` for details/usage.

    Parameters
    -----------------
    classifier: object
        A classifier instance supporting sklearn API.

    detector : `BaseDetector`
        To compute `conf_score`. All iterative handlers require this.

    threshold : float, default=.4
        Samples with lower conf_score will be filtered out.

    eps : float, default=.99
        Stopping criterion for main detection->cleaning loop, indicates ratio \
        of total number of mislabelled samples identified in two successive \
        iterations.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility
    """
    def __init__(self, classifier, detector, threshold=.4, eps=.99,
                 n_jobs=1, random_state=None):
        super().__init__(classifier, detector, n_jobs=n_jobs, random_state=random_state)
        self.threshold = threshold
        self.eps = eps

    def clean(self, X, y):
        X, y, conf_score = self._check_everything(X, y, conf_score=None)
        Xt, yt = X.copy(), y.copy()
        while True:
            clean_idx = conf_score > self.threshold

            N = len(X) - len(Xt)  # size of A_(j-1) i.e. no of noisy instances detected so far
            Xa, ya = Xt[clean_idx], yt[clean_idx]

            # If new labels have fewer classes than original...
            if len(np.unique(y)) != len(np.unique(ya)):
                warnings.warn("One or more of the classes has been completely "
                              "filtered out, stopping iteration.")
                break
            else:
                Xt, yt = Xa, ya

            if len(X) == len(Xt):
                warnings.warn("No noisy sample found, stopping at first iteration")
                break

            if N / (len(X) - len(Xt)) >= self.eps:
                break

            conf_score = self.detector.detect(Xt, yt)

        return Xt, yt

    def fit(self, X, y, conf_score=None):
        if conf_score is not None:
            raise RuntimeWarning("conf_score will be ignored. Iterative handlers only use "
                                 "Detector passed during construction.")

        Xf, yf = self.clean(X, y)
        self.classifier.fit(Xf, yf)
        return self

    @property
    def iterative(self):  # Does this Handler call Detector multiple times?
        return True

# TODO: Throw this away? merge with CLNI?
class IPF(BaseHandler, ClassifierMixin):
    """
    Iteratively detects and filters out mislabelled samples unless \
    a stopping criterion is met. See :cite:`ipf07` for details/usage.

    Differs slightly from `CLNI` in terms of how stopping criterion is \
    implemented.

    Parameters
    -----------------
    classifier: object
        A classifier instance supporting sklearn API.

    detector : `BaseDetector`
        To compute `conf_score`. All iterative handlers require this.

    threshold : float, default=.4
        Samples with lower conf_score will be filtered out.

    eps : float, default=.99
        Stopping criterion for main detection->cleaning loop, indicates ratio \
        of total number of mislabelled samples identified in two successive \
        iterations.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility
    """
    def __init__(self, classifier, detector, n_estimator=5, max_iter=3,
                 n_jobs=1, random_state=None):
        super().__init__(classifier, detector, n_jobs=n_jobs, random_state=random_state)
        self.n_estimator = n_estimator
        self.max_iter = max_iter

    def clean(self, X, y):
        Xf, yf = shuffle(X, y, random_state=self.random_state)
        orig_size = len(X)
        n_iters_with_small_change = 0
        tmp = 0

        Xf, yf, conf_score = self._check_everything(Xf, yf, conf_score=None)
        while n_iters_with_small_change < self.max_iter:
            tmp += 1
            cur_size = len(Xf)

            clean_idx = conf_score > .5  # Idx of clean samples
            Xa, ya = Xf[clean_idx], yf[clean_idx]

            # If new labels have fewer classes than original...
            if len(np.unique(y)) != len(np.unique(ya)):
                warnings.warn("One or more of the classes has been completely "
                              "filtered out, stopping iteration.")
                break
            else:
                Xf, yf = Xa, ya

            conf_score = self.detector.detect(Xf, yf)  # Calling detect once more than necessary

            cur_change = cur_size - len(Xf)
            if cur_change <= .01 * orig_size:
                n_iters_with_small_change += 1
            else:
                n_iters_with_small_change = 0  # Because these small change has to be consecutively 3 times
        return Xf, yf

    # Duplicate fit, a IterativeHandlerBase?
    def fit(self, X, y, conf_score=None):
        if conf_score is not None:
            raise RuntimeWarning("conf_score will be ignored. Iterative handlers only use "
                                 "Detector passed during construction.")
        Xf, yf = self.clean(X, y)
        assert len(np.unique(y)) == len(np.unique(yf)), "One or more of the classes has been completely filtered out"
        self.classifier.fit(Xf, yf)
        return self

    @property
    def iterative(self):  # Does this Handler call Detector multiple times?
        return True
