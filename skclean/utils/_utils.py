from pathlib import Path
from time import ctime, perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, check_cv
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle, check_random_state

_intervals = (
    ('weeks', 604800),  # 60 * 60 * 24 * 7
    ('days', 86400),  # 60 * 60 * 24
    ('hours', 3600),  # 60 * 60
    ('minutes', 60),
    ('seconds', 1),
)


# Code taken from https://stackoverflow.com/a/24542445/4553309
def _display_time(seconds, granularity=4):
    if seconds < 60:
        return f"{seconds:.2f} seconds"

    result = []

    for name, count in _intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])


# TODO: Add support for downloading dataset
def load_data(dataset, stats=False):
    path = Path(__file__).parent.parent / f'datasets/{dataset}.csv'
    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file {dataset} does not exist")

    df = df.astype('float64')
    data = df.values
    X, Y = data[:, :-1], data[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    X = MinMaxScaler().fit_transform(X)
    if stats:
        labels, freq = np.unique(Y, return_counts=True)
        print(f"{dataset}, {X.shape}, {len(labels)}, {freq.min() / freq.max():.3f}\n")
    return shuffle(X, Y, random_state=42)


# TODO: Support resuming inside cross_val_score, use Dask?
def compare(models: dict, datasets: list, cv, df_path=None, n_jobs=-1,
            scoring='accuracy', random_state=None, verbose=True, **kwargs):
    """
    Compare different methods across several datasets, with support for \
    parallelization, reproducibility and automatic resumption. Output is \
    a csv file where each row represents a dataset and each column \
    represents a method/ algorithm. It's basically a wrapper around \
    `sklearn.model_selection.cross_val_score`- check this for more details.

    Note that support for resumption is somewhat limited, it can only \
    recover output of (dataset, method) pair for whom computation is fully \
    complete. In other words, if a 10-fold cross-validation is stopped \
    after 5-fold, the results of that 5-fold is lost.


    Parameters
    --------------
    models : dict
        Keys are model name, values are scikit-learn API compatible classifiers.

    datasets : list
        A list of either `string`, denoting dataset names to be loaded with \
        `load_data`, or a nested tuple of (name, (X, y)), denoting dataset \
        name, features and labels respectively.

    cv : int, cross-validation generator or an iterable
        if int, no of folds to use in stratified k-fold

    df_path : string, default=None
        Path to (csv) file to store results- will be overwritten if already \
        present.

    scoring : string, or a scorer callable object / function with signature \
     ``scorer(estimator, X, y)`` which should return only a single value.

    n_jobs : int, default=1
        No of parallel cpu cores to use

    random_state : int, default=None
        Set this value for reproducibility. Note that this will overwrite \
        existing random state of methods even if it's already present.

    verbose : Controls the verbosity level

    kwargs : Other parameters for ``cross_val_score``.

    """

    rns = check_random_state(random_state)
    cv = check_cv(cv)
    cv.random_state = rns.randint(100)
    seeds = iter(rns.randint(10 ** 8, size=len(models)*len(datasets)))

    try:
        df = pd.read_csv(df_path, index_col=0)
        if verbose:
            print("Result file found, resuming...")  # Word 'resuming' is used in test
    except (FileNotFoundError, ValueError):
        df = pd.DataFrame()

    for data in datasets:

        if type(data) == str:
            X, Y = load_data(data, stats=verbose)
        else:
            data, (X, Y) = data   # Nested tuple of (name, (data, target))

        if data not in df.index:
            df.loc[data, :] = None

        for name, clf in models.items():
            if 'n_jobs' in clf.get_params():
                clf.set_params(n_jobs=1)
            if 'random_state' in clf.get_params():
                clf.set_params(random_state=next(seeds))

            if name not in df.columns:
                df[name] = None

            if not pd.isna(df.loc[data, name]):
                v = df.loc[data, name]
                if verbose:
                    print(f"Skipping {data},{name} :{v}")
                continue
            elif verbose:
                print(f"Starting: {data}, {name} at {ctime()[-14:-4]}")

            start = perf_counter()
            res = cross_val_score(clf, X, Y, cv=cv, n_jobs=n_jobs,
                                  scoring=scoring, error_score='raise', **kwargs)
            df.at[data, name] = res.mean()
            elapsed_time = _display_time(perf_counter() - start)
            if verbose:
                print(f"Result: {df.loc[data, name]:.4f} in {elapsed_time} \n")
            if df_path:
                df.to_csv(df_path)
        if verbose:
            print()
    return df
