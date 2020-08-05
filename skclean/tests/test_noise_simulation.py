import pytest
from numpy.testing import assert_array_equal
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from skclean.simulate_noise import flip_labels_uniform
from skclean.simulate_noise import flip_labels_cc
from skclean.simulate_noise import UniformNoise
from skclean.simulate_noise import CCNoise, BCNoise
from skclean.utils.noise_generation import gen_simple_noise_mat
from skclean.utils.noise_generation import generate_noise_matrix_from_trace


SEED = 42
X, y = make_classification(100, 4)  # Must be multiple of 100


@pytest.mark.parametrize('nl', [.2, .4, .7])
def test_uniform_exact(nl):
    yn = flip_labels_uniform(y, nl, exact=True)
    assert (y != yn).sum() / len(y) == nl

    _, ya = UniformNoise(nl, random_state=SEED, exact=True).simulate_noise(X, y)
    assert (y != ya).sum() / len(y) == nl


@pytest.mark.parametrize('nl', [.2, .4, .7])
def test_random_state_uniform(nl):
    y1 = flip_labels_uniform(y, nl, random_state=SEED, exact=False)
    y2 = flip_labels_uniform(y, nl, random_state=SEED, exact=False)
    assert_array_equal(y1, y2)

    _, ya = UniformNoise(nl, random_state=SEED, exact=False).simulate_noise(X, y)
    _, yb = UniformNoise(nl, random_state=SEED, exact=False).simulate_noise(X, y)
    assert_array_equal(ya, yb)
    assert_array_equal(ya, y1)


@pytest.mark.parametrize('K', [2, 4, 10])
@pytest.mark.parametrize('nl', [.2, .4])
def test_random_state_cc(K, nl):
    lcm1 = gen_simple_noise_mat(K, nl, random_state=SEED)
    lcm2 = gen_simple_noise_mat(K, nl, random_state=SEED)
    assert_array_equal(lcm1, lcm2)

    y1 = flip_labels_cc(y, lcm1, random_state=SEED)
    y2 = flip_labels_cc(y, lcm2, random_state=SEED)
    _, yn = CCNoise(lcm1, random_state=SEED).simulate_noise(X, y)
    assert_array_equal(y1, y2)
    assert_array_equal(y1, yn)


@pytest.mark.parametrize('clf', [GaussianNB(), KNeighborsClassifier()])
@pytest.mark.parametrize('nl', [.2, .4])
def test_random_state_bc(clf, nl):
    _, y1 = BCNoise(clf, nl, random_state=SEED).simulate_noise(X, y)
    _, y2 = BCNoise(clf, nl, random_state=SEED).simulate_noise(X, y)
    assert_array_equal(y1, y2)


@pytest.mark.parametrize('clf', [DecisionTreeClassifier(max_depth=4)])
@pytest.mark.parametrize('nl', [.2, .4])
def test_bc_noise(clf, nl):
    _, yn = BCNoise(clf, nl).simulate_noise(X, y)
    assert (y != yn).sum() / len(y) == nl


if __name__=='__main__':
    test_uniform_exact(.2)