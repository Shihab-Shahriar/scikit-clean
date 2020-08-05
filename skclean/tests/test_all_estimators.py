# Check work with list, pandas

import pytest
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator, clone
from sklearn.utils.estimator_checks import check_estimators_pickle

from skclean.tests.common_stuff import (ALL_ESTIMATORS, DATASETS,
                                        NOISE_SIMULATORS, ALL_COMBS,
                                        NOISE_DETECTORS, NOISE_HANDLERS,
                                        ROBUST_MODELS)


# TODO: Check total test coverage

@pytest.mark.parametrize('est', ALL_ESTIMATORS)
@pytest.mark.parametrize('feature, target', DATASETS)
def test_data_types(est: BaseEstimator, feature, target):
    if hasattr(est, 'fit'):  # Meaning a Handler or Robust Model
        est.fit(feature, target).predict(feature)
    elif hasattr(est, 'detect'):
        est.detect(feature, target)
    elif hasattr(est, 'simulate_noise'):
        est.simulate_noise(feature, target)
    else:
        raise Exception("WTF")


NAMED_ESTIMATORS = [(c.__class__.__name__, c) for c in ALL_ESTIMATORS]


@pytest.mark.parametrize('name, est', NAMED_ESTIMATORS)
def test_pickling(name, est):
    if hasattr(est, 'fit'):
        check_estimators_pickle('b', est)


@pytest.mark.parametrize('name, est',
                         [(c.__class__.__name__, c) for c in NOISE_SIMULATORS])
@pytest.mark.parametrize('X, y', DATASETS)
def test_random_state_sim(name, est, X, y):
    est2 = clone(est)

    est.set_params(random_state=42)
    _, y1 = est.simulate_noise(X, y)

    est2.set_params(random_state=42)
    _, y2 = est2.simulate_noise(X, y)

    assert_array_equal(y1, y2)


@pytest.mark.parametrize('name, est',
                         [(c.__class__.__name__, c) for c in NOISE_DETECTORS])
@pytest.mark.parametrize('X, y', DATASETS)
def test_random_state_det(name, est, X, y):
    if 'random_state' not in est.get_params():
        return
    est2 = clone(est)

    est.set_params(random_state=42)
    conf_score1 = est.detect(X, y)

    est2.set_params(random_state=42)
    conf_score2 = est2.detect(X, y)

    assert_array_equal(conf_score1, conf_score2)

print(ALL_COMBS[44])

@pytest.mark.parametrize('name, est',
                         [(c.__class__.__name__, c) for c in ALL_COMBS])
@pytest.mark.parametrize('X, y', DATASETS)
def test_random_state_handl(name, est, X, y):
    if 'random_state' not in est.get_params():
        return

    est2 = clone(est)

    est.set_params(random_state=42)
    y1 = est.fit(X, y).predict(X)

    est2.set_params(random_state=42)
    y2 = est2.fit(X, y).predict(X)

    assert_array_equal(y1, y2)


@pytest.mark.parametrize('name, est',
                         [(c.__class__.__name__, c) for c in ROBUST_MODELS])
@pytest.mark.parametrize('X, y', DATASETS)
def test_random_state_models(name, est, X, y):
    if 'random_state' not in est.get_params():
        return
    est2 = clone(est)

    est.set_params(random_state=42)
    y1 = est.fit(X, y).predict(X)

    est2.set_params(random_state=42)
    y2 = est2.fit(X, y).predict(X)

    assert_array_equal(y1, y2)

# X,y = DATASETS[2]
# c = ROBUST_MODELS[1].fit(X,y).predict(X)
# print("DONE")
