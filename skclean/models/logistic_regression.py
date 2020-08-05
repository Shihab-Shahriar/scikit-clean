"""Two algorithms: NIPS'14 & ECML'12"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import log_logistic
from sklearn.utils.multiclass import unique_labels


def log_loss(wp, X, target, C, PN, NP):
    """
    It is minimized using "L-BFGS-B" method of "scipy.optimize.minimize" function, and results in
    similar coefficients as sklearn's Logistic Regression when PN=NP=0.0.

    Parameters
    -------------
    wp: Coefficients & Intercept
    X: (N,M) shaped data matrix
    target: (N,) shaped 1-D array of targets
    C: Regularization
    PN: % of Positive samples labeled as Negative
    NP: % of Positive samples labeled as Negative

    Returns
    ------------
    loss_value: float

    """
    c = wp[-1]
    w = wp[:-1]
    z = np.dot(X, w) + c
    yz = target * z  # to compute l(t,y)
    nyz = -target * z  # to compute l(t,-y)
    ls = -log_logistic(yz)  # l(t,y)
    nls = -log_logistic(nyz)  # l(t,-y)
    idx = target == 1  # indexes of samples w/ P label
    loss = ls.copy()  # To store l-hat
    loss[idx] = (1 - NP) * ls[idx] - PN * nls[idx]  # Modified loss for P samples
    loss[~idx] = (1 - PN) * ls[~idx] - NP * nls[~idx]  # Modified loss for N samples
    loss = loss / (1 - PN - NP) + .5 * (1. / C) * np.dot(w, w)  # Normalization & regularization
    return loss.sum()  # Final loss


class RobustLR(LogisticRegression):
    """
    Modifies the logistic loss using class dependent (estimated) noise rates \
    for robustness. This implementation is for binary classification tasks only.

    See :cite:`natarajan13` for details.

    Parameters
    ----------------
    PN : float, default=.2
        Percentage of Positive labels flipped to Negative.

    NP : float, default=.2
        Percentage of Negative labels flipped to Positive.

    C : float
        Inverse of regularization strength, must be a positive float.

    random_state : int, default=None
        Set this value for reproducibility
    """

    def __init__(self, PN=.2, NP=.2, C=np.inf, max_iter=4000, random_state=None):
        super().__init__(C=C, max_iter=max_iter, random_state=random_state)
        self.PN = PN
        self.NP = NP

    # TODO: Support `sample_weight`
    def fit(self, X, y,  sample_weight=None):
        X, y = self._validate_data(X, y)

        self.classes_ = unique_labels(y)
        w0 = np.zeros(X.shape[1] + 1)
        target = y.copy()
        target[target == 0] = -1
        self.r_ = minimize(log_loss, w0, method="L-BFGS-B", args=(X, target, self.C, self.PN, self.NP),
                           options={"maxiter": self.max_iter})
        self.coef_ = self.r_.x[:-1].reshape(1, -1)
        self.intercept_ = self.r_.x[-1:]
        return self
