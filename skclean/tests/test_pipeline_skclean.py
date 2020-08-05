import pickle
import pytest
from numpy.testing import assert_allclose
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import cross_val_score, GridSearchCV, ShuffleSplit

from skclean.detectors.neighbors import KDN
from skclean.detectors.ensemble import RandomForestDetector
from skclean.handlers.filters import Filter
from skclean.pipeline import Pipeline, make_pipeline
from skclean.simulate_noise import UniformNoise, CCNoise


X, y = load_iris(return_X_y=True)
lcm = [[.7, .1, .2],
       [.3, .6, .1],
       [.1, .21, .69]]
CV1 = ShuffleSplit(n_splits=1, test_size=.23)


@pytest.mark.parametrize('sim', [UniformNoise(.3), CCNoise(lcm)])
def test_noise_sim(sim):
    p = Pipeline([
        ('s', sim),
        ('c', DecisionTreeClassifier())
    ])
    p.fit(X, y)
    p.predict(X)

    p = make_pipeline(sim, DecisionTreeClassifier())
    p.fit(X, y)
    p.predict(X)

def test_stack_sim():
    p = Pipeline([
        ('a', UniformNoise(.3)),
        ('b', CCNoise(lcm)),
        ('c', DecisionTreeClassifier()),
    ])
    p.fit(X, y)
    p.predict(X)

def test_detector_init():
    p = Pipeline([
        ('a', UniformNoise(.3)),
        ('p', StandardScaler()),
        ('d', RandomForestDetector(n_estimators=50)),
        ('c', Filter(DecisionTreeClassifier()))
    ])
    p.fit(X, y)
    p.predict(X)

def test_pipe_cv():
    p = Pipeline([
        ('a', UniformNoise(.3)),
        ('p', StandardScaler()),
        ('d', KDN(n_neighbors=5)),
        ('c', Filter(DecisionTreeClassifier()))
    ])
    p = cross_val_score(p, X, y, cv=CV1, error_score='raise')
    print(p)

def test_pipe_grid():
    params = {'p__n_components': [2, 4],
              'd__n_neighbors': [2, 5],
              'c__threshold': [.3, .5],
              'c__classifier__max_depth': [2, 4]}
    p = Pipeline([
        ('a', CCNoise(lcm)),
        ('s', StandardScaler()),
        ('p', PCA()),
        ('d', KDN()),
        ('c', Filter(DecisionTreeClassifier()))
    ])
    g = GridSearchCV(p, params, cv=CV1, error_score='raise')
    g.fit(X, y)
    g.predict(X)

@pytest.mark.parametrize('X,y',[
    load_iris(return_X_y=True),
    make_classification(100, 4)
])
def test_pipe_pickling(X, y):
    p = Pipeline([
        ('a', CCNoise(lcm)),
        ('s', StandardScaler()),
        ('p', PCA()),
        ('d', KDN()),
        ('c', Filter(DecisionTreeClassifier()))
    ])
    p.fit(X, y)
    q = pickle.loads(pickle.dumps(p))
    assert_allclose(p.predict(X), q.predict(X))

@pytest.mark.xfail(raises=TypeError, reason="Detector used without a Handler after it")
def test_det_wo_hand():
    p = Pipeline([
        ('a', CCNoise(lcm)),
        ('s', StandardScaler()),
        ('p', PCA()),
        ('d', KDN()),
        ('c', DecisionTreeClassifier())
    ])
    p.fit(X, y)
    p.predict(X)


@pytest.mark.xfail(raises=ValueError, reason="Handler used in Pipeline without a Detector before it")
def test_hand_wo_det():
    p = Pipeline([
        ('a', CCNoise(lcm)),
        ('s', StandardScaler()),
        ('p', PCA()),
        ('c', Filter(DecisionTreeClassifier()))
    ])
    p.fit(X, y)
    p.predict(X)