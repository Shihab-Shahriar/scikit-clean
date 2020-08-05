Scikit-clean
==================

**scikit-clean** is a python ML library for classification in the presence of \
label noise. Aimed primarily at researchers, this provides implementations of \
several state-of-the-art algorithms; tools to simulate artificial noise, create complex pipelines \
and evaluate them.

This library is fully scikit-learn API compatible: which means \
all scikit-learn's building blocks can be seamlessly integrated into workflow. \
Like scikit-learn estimators, most of the methods also support features like \
parallelization, reproducibility etc.

Example Usage
***************
A typical label noise research workflow begins with clean labels, simulates \
label noise into training set, and then evaluates how a model handles that noise \
using clean test set. In scikit-clean, this looks like:

.. code-block:: python

    from skclean.simulate_noise import flip_labels_uniform
    from skclean.models import RobustLR   # Robust Logistic Regression

    X, y = make_classification(n_samples=200,n_features=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

    y_train_noisy = flip_labels_uniform(y_train, .3)  # Flip labels of 30% samples
    clf = RobustLR().fit(X_train,y_train_noisy)
    print(clf.score(X_test, y_test))

scikit-clean provides a customized `Pipeline` for more complex workflow. Many noise robust \
algorithms can be broken down into two steps: detecting noise likelihood for each sample
in the dataset, and train robust classifiers by using that information. This fits
nicely with Pipeline's API:

.. code-block:: python

    # ---Import scikit-learn stuff----
    from skclean.simulate_noise import UniformNoise
    from skclean.detectors import KDN
    from skclean.handlers import Filter
    from skclean.pipeline import Pipeline, make_pipeline  # Importing from skclean, not sklearn


    clf = Pipeline([
            ('scale', StandardScaler()),          # Scale features
            ('feat_sel', VarianceThreshold(.2)),  # Feature selection
            ('detector', KDN()),                  # Detect mislabeled samples
            ('handler', Filter(SVC())),           # Filter out likely mislabeled samples and then train a SVM
    ])

    clf_g = GridSearchCV(clf,{'detector__n_neighbors':[2,5,10]})
    n_clf_g = make_pipeline(UniformNoise(.3),clf_g)  # Create label noise at the very first step

    print(cross_val_score(n_clf_g, X, y, cv=5).mean())  # 5-fold cross validation

Please see this notebook_ before you begin for a more detailed introduction, \
and this_ for complete API.

.. _notebook: examples/Introduction%20to%20Scikit-clean.html
.. _this: api.html

Installation
******************

Simplest option is probably using pip::

    pip install scikit-clean

If you intend to modify the code, install in editable mode::

    git clone https://github.com/Shihab-Shahriar/scikit-clean.git
    cd scikit-clean
    pip install -e .

If you're only interested in small part of this library, say one or two algorithms, feel free to simply \
copy/paste relevant code into your project.

Alternatives
**************
There are several open source tools to handle label noise, some of them are: \

1. Cleanlab_
2. Snorkel_
3. NoiseFiltersR_

.. _Cleanlab: https://github.com/cgnorthcutt/cleanlab
.. _Snorkel: https://github.com/snorkel-team/snorkel
.. _NoiseFiltersR: https://journal.r-project.org/archive/2017/RJ-2017-027/RJ-2017-027.pdf

`NoiseFiltersR` is closest in objective as ours, though it's implemented in R, and doesn't \
appear to be actively maintained.

`Cleanlab` and `Snorkel` are both in Python, though they have somewhat different \
priorities than us. While our goal is to implement as many algorithms as \
possible, these tools usually focus on one or few related papers. They have also been \
developed for some time- meaning they are more stable, well-optimized and better suited \
for practitioners/ engineers than `scikit-clean`.



Credits
**************

We want to `scikit-learn`, `imbalance-learn` and `Cleanlab`, these implemntations \
are inspired by, and dircetly borrows code from these libraries.

We also want to thank the authors of original papers. Here is a list of papers partially \
or fully implemented by `scikit-clean`:

.. bibliography:: zrefs.bib
    :list: bullet
    :cited:

A note about naming
-----------------------------------------------

    "There are 2 hard problems in computer science: cache invalidation, naming things, and \
    off-by-1 errors."

Majority of the algorithms in `scikit-clean` are not explicitly named by their authors. \
In some rare cases, similar or very similar ideas appear under different names (e.g. `KDN`). \
We tried to name things as best as we could. However, if you're the author of any of these \
methods and want to rename it, we'll happily oblige.




