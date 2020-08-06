# Test if example code in readme file runs properly

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC


def test_example_one():
    from skclean.simulate_noise import flip_labels_uniform
    from skclean.models import RobustLR  # Robust Logistic Regression

    X, y = make_classification(n_samples=200, n_features=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

    y_train_noisy = flip_labels_uniform(y_train, .3)  # Flip labels of 30% samples
    clf = RobustLR().fit(X_train, y_train_noisy)
    print(clf.score(X_test, y_test))


def test_example_two():
    from skclean.simulate_noise import UniformNoise
    from skclean.detectors import KDN
    from skclean.handlers import Filter
    from skclean.pipeline import Pipeline, make_pipeline  # Importing from skclean, not sklearn
    from skclean.utils import load_data

    X, y = load_data('breast_cancer')

    clf = Pipeline([
        ('scale', StandardScaler()),  # Scale features
        ('feat_sel', VarianceThreshold(.2)),  # Feature selection
        ('detector', KDN()),  # Detect mislabeled samples
        ('handler', Filter(SVC())),  # Filter out likely mislabeled samples and then train a SVM
    ])

    clf_g = GridSearchCV(clf, {'detector__n_neighbors': [2, 5, 10]})
    n_clf_g = make_pipeline(UniformNoise(.3), clf_g)  # Create label noise at the very first step

    print(cross_val_score(n_clf_g, X, y, cv=5).mean())  # 5-fold cross validation


if __name__ == '__main__':
    test_example_one()
    test_example_two()
