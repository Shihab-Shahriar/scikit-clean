import numpy as np
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import minmax_scale
from sklearn.utils import check_random_state

from skclean.utils.noise_generation import gen_simple_noise_mat


def _flip_idx(Y, target_idx, random_state=None):
    """Flip the labels of `target_idx` to random label"""
    labels = np.unique(Y)
    random_state = check_random_state(random_state)

    target_mask = np.full(Y.shape, 0, dtype=np.bool)
    target_mask[target_idx] = 1
    yn = Y.copy()
    mask = target_mask.copy()
    while True:
        left = mask.sum()
        if left == 0:
            break
        new_labels = random_state.choice(labels, size=left)
        yn[mask] = new_labels
        mask = mask & (yn == Y)
    return yn


def flip_labels_uniform(Y: np.ndarray, noise_level: float, *, random_state=None, exact=True):
    """
    All labels are equally likely to be flipped, irrespective of their true \
    label or feature. The new (noisy) label is also chosen with uniform \
    probability from alternative class labels.

    Parameters
    -----------------
    Y: np.ndarray
        1-D array of labels

    noise_level: float
        percentage of labels to flip

    random_state : int, default=None
        Set this value for reproducibility

    exact: bool default=True
        If True, the generated noise will be as close to `noise_level` as possible.
        The approximate version (i.e. exact=False) is faster but less accurate.

    Returns
    -----------
    Yn: np.ndarray
        1-D array of flipped labels
    """

    if not exact:
        labels = np.unique(Y)
        n_labels = len(labels)
        lcm = np.full((n_labels, n_labels), noise_level / (n_labels - 1))
        np.fill_diagonal(lcm, 1 - noise_level)
        return flip_labels_cc(Y, lcm, random_state=random_state)

    random_state = check_random_state(random_state)
    nns = int(len(Y) * noise_level)
    target_idx = random_state.choice(len(Y), size=nns, replace=False)
    yn = _flip_idx(Y, target_idx, random_state=random_state)
    assert (yn[target_idx] == Y[target_idx]).sum() == 0
    return yn


# TODO: create an *exact* version of this
def flip_labels_cc(y, lcm, random_state=None):
    """
    Class Conditional Noise: general version of flip_labels_uniform, a \
    sample's probability of getting mislabelled and it's new (noisy) \
    label depends on it's true label, but not features.

    Parameters
    -----------------
    Y: np.ndarray
        1-D array of labels

    lcm: np.ndarray
        Short for Label Confusion Matrix. `lcm[i,j]` denotes the probability \
        of a sample with true label `i` getting mislabelled as `j`.

    random_state : int, default=None
        Set this value for reproducibility

    Returns
    -----------
    Yn: np.ndarray
        1-D array of flipped labels
    """

    lcm = np.array(lcm)
    lcm = lcm / lcm.sum(axis=1).reshape(-1, 1)  # Each row sums to 1
    a = lcm[y]
    s = a.cumsum(axis=1)

    random_state = check_random_state(random_state)
    r = random_state.rand(a.shape[0])[:, None]
    yn = (s > r).argmax(axis=1)
    return yn


# -----------------------------------------------------------------

class NoiseSimulator(BaseEstimator, TransformerMixin):

    def __init__(self, random_state=None):
        self.random_state = random_state

    def simulate_noise(self, X, y):
        raise NotImplementedError("Attempt to instantiate abstract class")

    def fit_transform(self, X, y=None, **fit_params):
        return self.simulate_noise(X, y)

    def transform(self, X):
        return X


class UniformNoise(NoiseSimulator):
    """
    All labels are equally likely to be flipped, irrespective of their true \
    label or feature. The new (noisy) label is also chosen with uniform \
    probability from alternative class labels. Simple wrapper around \
    `flip_labels_uniform` mainly for use in `Pipeline`.

    Parameters
    -----------------
    noise_level: float
        percentage of labels to flip

    exact: bool default=True
        If True, the generated noise will be as close to `noise_level` as possible.
        The approximate version (i.e. exact=False) is faster but less accurate.

    random_state : int, default=None
        Set this value for reproducibility
    """

    def __init__(self, noise_level, exact=True, random_state=None):
        super().__init__(random_state=random_state)
        self.noise_level = noise_level
        self.exact = exact

    def simulate_noise(self, X, y):
        X, y = self._validate_data(X, y)
        yn = flip_labels_uniform(y, self.noise_level, random_state=self.random_state,
                                 exact=self.exact)
        return X, yn


class CCNoise(NoiseSimulator):
    """
    Class Conditional Noise: general version of `flip_labels_uniform`- \
    a sample's probability of getting mislabelled and it's new (noisy) \
    label depends on it's true label, but not features. Simple wrapper \
    around `flip_labels_cc` mainly for use in `Pipeline`.

    Parameters
    -----------------
    lcm: np.ndarray
        Short for Label Confusion Matrix. `lcm[i,j]` denotes the probability \
        of a sample with true label `i` getting mislabelled as `j`.

    random_state : int, default=None
        Set this value for reproducibility
    """

    def __init__(self, lcm=None, random_state=None):
        super().__init__(random_state=random_state)
        self.lcm = lcm

    def simulate_noise(self, X, y):
        lcm = self.lcm
        if self.lcm is None or isinstance(self.lcm, float):
            noise_level = self.lcm or .2
            K = len(np.unique(y))
            lcm = gen_simple_noise_mat(K, noise_level, self.random_state)

        X, y = self._validate_data(X, y)
        yn = flip_labels_cc(y, lcm, self.random_state)
        return X, yn


class BCNoise(NoiseSimulator):
    """
    Boundary Consistent Noise- instances closer to boundary more likely to \
    be noisy. In this implementation, "closeness" to decision boundary of a \
    sample is measured using entropy of it's class probabilities. A
    classifier with support for well calibrated class probabilities (i.e. \
    `predict_proba` of scikit-learn API) is required.

    Only supports binary classification for now. See :cite:`idnoise18` for \
    details.

    Parameters
    -------------------
    classifier : object
        A classifier instance supporting sklearn API.

    noise_level: float
        percentage of labels to flip

    random_state : int, default=None
        Set this value for reproducibility

    """

    def __init__(self, classifier, noise_level, random_state=None):
        self.classifier = classifier
        self.noise_level = noise_level
        self.random_state = random_state

    def simulate_noise(self, X, y):
        X, y = self._validate_data(X, y)

        rns = check_random_state(self.random_state)
        if 'random_state' in self.classifier.get_params():
            self.classifier.set_params(random_state=rns.randint(10**3))

        probs = self.classifier.fit(X, y).predict_proba(X)
        e = entropy(probs, axis=1) + .01  # Otherwise 0-entropy samples would never be selected
        e = e / e.max()
        nns = int(len(y) * self.noise_level)
        target_idx = rns.choice(len(y), size=nns, replace=False, p= e/e.sum())
        return X, _flip_idx(y, target_idx, random_state=self.random_state)
