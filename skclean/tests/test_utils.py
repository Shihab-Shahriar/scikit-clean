import os
import tempfile

import pytest
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.datasets import make_classification
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from skclean.utils import compare


m = {
    'dt': DecisionTreeClassifier(max_depth=4),
    'knn': KNeighborsClassifier(),
}
ds = ['iris', ('make', make_classification(100, 4))]
cv = ShuffleSplit(n_splits=1, test_size=.2)


@pytest.fixture
def tmp_path():
    path = os.path.join(tempfile.gettempdir(), 'tmp.csv')
    yield path
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def test_compare_random_state():
    df1 = compare(m, ds, cv, df_path=None, n_jobs=2, random_state=42)
    df2 = compare(m, ds, cv, df_path=None, n_jobs=2, random_state=42)

    assert df1.equals(df2)


def test_compare_resumption(capsys, tmp_path):
    df = compare(m, ds, cv, df_path=tmp_path, n_jobs=2, random_state=42)

    # Simulate stoppage mid-computation
    df1 = df.copy()
    for i in range(2):
        row = np.random.choice(df1.index)
        col = np.random.choice(df1.columns)
        df1.loc[row, col] = None
    df1.to_csv(tmp_path)

    df2 = compare(m, ds, cv, df_path=tmp_path, n_jobs=2,
                  random_state=42, verbose=True)

    assert_frame_equal(df, df2, check_dtype=False)
    stdout = capsys.readouterr().out
    assert 'resuming' in stdout

    # Check resumption when new method is added
    m['nb'] = GaussianNB()
    df3 = compare(m, ds, cv, df_path=tmp_path, n_jobs=2,
                  random_state=42, verbose=True)
    stdout = capsys.readouterr().out
    assert 'resuming' in stdout
    assert_frame_equal(df, df3.drop(columns=['nb']), check_dtype=False)
