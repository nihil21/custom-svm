import numpy as np
from typing import Optional


class SVM:
    """Class implementing a Support Vector Machine: instead of minimising the primal function
        L_P(w, b, lambda_mat) = 1/2 ||w||^2 - sum_i{lambda_i[(w * x + b) - 1]},
    the dual function
        L_D(lambda_mat) = sum_i{lambda_i} - 1/2 sum_i{sum_j{lambda_i lambda_j y_i y_j K(x_i, x_j)}}
    is maximised.

    Attributes:
        lambda_mat --- matrix of lagrangian multipliers
        support_vectorsP --- list of the support vectors belonging to the positive category
        support_vectorsN --- list of the support vectors belonging to the negative category
        gamma --- parameter of the kernel function
        kernel --- kernel function
        is_fit --- boolean variable indicating whether the SVM is fit or not"""

    def __init__(self, kernel: Optional[str] = 'rbf', gamma: Optional[float] = None):
        self.lambda_mat = []
        self.support_vectorsP = []
        self.support_vectorsN = []
        # Gamma will be computed during fit process
        self.gamma = gamma
        # RBF used as kernel
        self.kernel = lambda x_i, x_j: np.exp(-self.gamma * np.inner(x_i - x_j))
        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        # If gamma was not specified in '__init__', it is set according to the 'scale' approach
        if not self.gamma:
            self.gamma = 1/(n_features * X.var())


        self.is_fit = True
