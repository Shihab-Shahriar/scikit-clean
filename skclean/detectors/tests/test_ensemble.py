# Contains tests for one or specific/narrow use cases
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from skclean.detectors.ensemble import InstanceHardness
from skclean.detectors.ensemble import RandomForestDetector

X, y = make_classification(100, 4)


@pytest.mark.parametrize('det', [RandomForestClassifier(n_estimators=10),
                                 KNeighborsClassifier()])
def test_single_clf_IH(det):
    InstanceHardness(classifiers=det).fit_transform(X, y)


@pytest.mark.parametrize('method', ['cv', 'bootstrap'])
def test_rf_detector(method):
    RandomForestDetector(method=method).fit_transform(X, y)
