import warnings

import numpy as np
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils import check_random_state

from .base import BaseDetector


class PartitioningDetector(BaseDetector):
    """
    Partitions dataset into n subsets, trains a classifier on each.
    Trained models are then used to predict on entire dataset.

    See :cite:`ipf07` for details.

    Parameters
    ------------
    classifier : object, default=None
        A classifier instance supporting sklearn API.
        If None, `DecisionTreeClassifier` is used.

    n_partitions : int, default=5
        No of non-overlapping partitions created from dataset.
        For small datasets, you might want to use smaller values.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility

    """

    def __init__(self, classifier=None, n_partitions=5, n_jobs=1, random_state=None):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.classifier = classifier
        self.n_partitions = n_partitions

    def detect(self, X, y):
        X, y = self._check_everything(X, y)

        classifier = clone(self.classifier) if self.classifier else \
            DecisionTreeClassifier(max_depth=2, random_state=self.random_state)

        breaks = [(len(X) // self.n_partitions) * i
                  for i in range(1, self.n_partitions)]
        Xs, ys = np.split(X, breaks), np.split(y, breaks)

        clfs = []
        for i in range(self.n_partitions):  # All clfs have same random_state but diff data
            c = clone(classifier).fit(Xs[i], ys[i])
            clfs.append(c)

        preds = np.zeros((len(X), self.n_partitions))
        for i in range(self.n_partitions):
            preds[:, i] = clfs[i].predict(X)
        eqs = preds == y.reshape(-1, 1)

        return eqs.sum(axis=1) / self.n_partitions


class MCS(BaseDetector):
    """
    Detects noise using a sequential Markov Chain Monte Carlo sampling algorithm.
    Tested for binary classification, multi-class classification sometimes
    perform poorly. See :cite:`mcmc19` for details.

    Parameters
    --------------
    classifier : object, default=None
        A classifier instance supporting sklearn API.
        If None, `LogisticRegression` is used.

    n_steps : int, default=20
        No of sampling steps to run.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility

    """

    def __init__(self, classifier=None, n_steps=20, n_jobs=1, random_state=None):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.classifier = classifier
        self.n_steps = n_steps

    def detect(self, X, y):
        X, y = self._check_everything(X, y)
        rns = check_random_state(self.random_state)
        seeds = rns.randint(10 ** 8, size=self.n_steps)

        classifier = clone(self.classifier) if self.classifier \
            else LogisticRegression(random_state=self.random_state)
        contain_random_state = 'random_state' in classifier.get_params()

        mask = np.ones(y.shape, 'bool')
        conf_score = np.zeros(y.shape)
        for i in range(self.n_steps):
            conf_score[mask] += 1

            clf = clone(classifier)
            if contain_random_state:
                clf.set_params(random_state=seeds[i])

            clf.fit(X[mask], y[mask])
            probs = clf.predict_proba(X)  # (N,n_estimators), p(k|x) for all k in classes
            pc = probs[range(len(y)), y]  # (N,), Prob assigned to correct class

            mask = rns.binomial(1, pc).astype('bool')

            if not np.all(np.unique(y[mask]) == self.classes_):
                warnings.warn(f"One or more classes have been entirely left out "
                              f"in current iteration {i}, stopping MCMC loop.",
                              category=RuntimeWarning)
                break

        return conf_score / self.n_steps


class InstanceHardness(BaseDetector):
    """
    A set of classifiers are used to predict labels of each sample
    using cross-validation. `conf_score` of a sample is percentage
    classifiers that correctly predict it's label. See :cite:`ih14`
    for details.

    Parameters
    --------------
    classifiers : list, default=None
        Classifiers used to predict sample labels.
        If None, four classifiers are used: `GaussianNB`,
        `DecisionTreeClassifier`, `KNeighborsClassifier` and `LogisticRegression`.

    cv : int, cross-validation generator or an iterable, default=None
        If None, uses 5-fold stratified k-fold
        if int, no of folds to use in stratified k-fold

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility

    """

    DEFAULT_CLFS = [DecisionTreeClassifier(max_leaf_nodes=500), GaussianNB(), KNeighborsClassifier(),
                    LogisticRegression(multi_class='auto', max_iter=4000, solver='lbfgs')]

    def __init__(self, classifiers=None, cv=None, n_jobs=1, random_state=None):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.classifiers = classifiers
        self.cv = cv

    def detect(self, X, y):
        X, y = self._check_everything(X, y)

        if self.classifiers is None:
            self.classifiers = InstanceHardness.DEFAULT_CLFS

        cv = self.cv
        if cv is None or type(cv) == int:
            n_splits = self.cv or 5
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=self.random_state)

        N = len(X)
        conf_score = np.zeros_like(y, dtype='float64')
        rns = check_random_state(self.random_state)
        seeds = rns.randint(10 ** 8, size=len(self.classifiers))
        for i, clf in enumerate(self.classifiers):
            if 'random_state' in clf.get_params():
                clf.set_params(random_state=seeds[i])

            # probability given to original class of all samples
            probs = cross_val_predict(clf, X, y, cv=cv, n_jobs=self.n_jobs,
                                      method='predict_proba')[range(N), y]

            conf_score += probs

        return conf_score / len(self.classifiers)


class RandomForestDetector(BaseDetector):
    """
    Trains a Random Forest first- for each sample, only trees that
    didn't select it for training (via bootstrapping) are used to
    predict it's label. Percentage of trees that correctly predicted
    the label is the sample's `conf_score`.

    See :cite:`twostage18` for details.

    n_estimators : int, default=101
        No of trees in Random Forest.

    sampling_ratio : float, 0.0 to 1.0, default=1.0
        No of samples drawn at each tree equals: len(X) * sampling_ratio

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility
    """

    # TODO: Allow other tree ensembles
    def __init__(self, n_estimators=101, sampling_ratio=None, n_jobs=1, random_state=None):
        super().__init__(n_jobs=n_jobs, random_state=random_state)
        self.n_estimators = n_estimators
        self.sampling_ratio = sampling_ratio

    def detect(self, X, y):
        X, y = self._validate_data(X, y)

        rf = RandomForestClassifier(n_estimators=self.n_estimators, oob_score=True,
                                    max_samples=self.sampling_ratio, n_jobs=self.n_jobs,
                                    random_state=self.random_state).fit(X, y)

        conf_score = rf.oob_decision_function_[range(len(X)), y]
        return conf_score
