import numpy as np
import itertools
import cvxopt
from typing import Optional


class SVM:
    """Class implementing a Support Vector Machine: instead of minimising the primal function
        L_P(w, b, lambda_mat) = 1/2 ||w||^2 - sum_i{lambda_i[(w * x + b) - 1]},
    the dual function
        L_D(lambda_mat) = sum_i{lambda_i} - 1/2 sum_i{sum_j{lambda_i lambda_j y_i y_j K(x_i, x_j)}}
    is maximised.

    Attributes:
        kernel --- type of the kernel ['linear'/'rbf'/'poly'/'sigmoid']
        kernel_fn --- kernel function
        gamma --- parameter of the kernel function
        lambdas --- lagrangian multipliers
        sv_X --- support vectors related to X
        sv_y --- support vectors related to y
        w --- matrix of hyperplane parameters
        b --- hyperplane bias
        is_fit --- boolean variable indicating whether the SVM is fit or not"""

    def __init__(self,
                 kernel: Optional[str] = 'linear',
                 gamma: Optional[float] = None,
                 deg: Optional[int] = 3,
                 r: Optional[float] = 0.0):
        # Lagrangian multipliers, hyper-parameters and support vectors are initially set to None
        self.lambdas = None
        self.sv_X = None
        self.sv_y = None
        self.w = None
        self.b = None

        # If gamma is None, it will be computed during fit process
        self.gamma = gamma

        # Assign the right kernel
        self.kernel = kernel
        if kernel == 'linear':
            self.kernel_fn = lambda x_i, x_j: x_i.dot(x_j)
        elif kernel == 'rbf':
            self.kernel_fn = lambda x_i, x_j: np.exp(-self.gamma * np.inner(x_i - x_j, x_i - x_j))
        elif kernel == 'poly':
            self.kernel_fn = lambda x_i, x_j: (gamma * x_i.dot(x_j) + r) ** deg
        elif kernel == 'sigmoid':
            self.kernel_fn = lambda x_i, x_j: np.tanh(x_i.dot(x_j) + r)

        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        # If gamma was not specified in '__init__', it is set according to the 'scale' approach
        if not self.gamma:
            self.gamma = 1/(n_features * X.var())

        # max{L_D(Lambda)} can be rewritten as
        #   min{1/2 Lambda^T H Lambda - 1^T Lambda}
        #       s.t. -lambda_i <= 0
        #       s.t. y^t Lambda = 0
        # This form is conform to the signature of the quadratic solver provided by CVXOPT library:
        #   min{1/2 x^T P x + q^T x}
        #       s.t. G x <= h
        #       s.t. A x = b
        # where P is an n_samples*n_samples matrix, where P[i][j] = y_i y_j K(x_i, x_j)
        K = np.zeros(shape=(n_samples, n_samples))
        for i, j in itertools.product(range(n_samples), range(n_samples)):
            K[i, j] = self.kernel_fn(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        G = cvxopt.matrix(-np.eye(n_samples))
        h = cvxopt.matrix(np.zeros(n_samples))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(np.zeros(1))

        # Compute the solution using the quadratic solver
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Extract Lagrange multipliers
        lambdas = np.ravel(sol['x'])
        # Find indices of the support vectors, which have non-zero Lagrange multipliers, and save the support vectors
        # as instance attributes
        sv_idx = lambdas > 1e-4
        self.sv_X = X[sv_idx]
        self.sv_y = y[sv_idx]
        self.lambdas = lambdas[sv_idx]
        print('{0:d} support vectors out of {1:d} points'.format(len(self.lambdas), n_samples))
        # Compute b
        ind = np.arange(len(lambdas))[sv_idx]
        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lambdas * self.sv_y[i] * K[ind[i], sv_idx])
        self.b /= len(self.lambdas)
        # Compute w
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for i in range(len(self.lambdas)):
                self.w[i] = self.lambdas[i] * self.sv_X[i] * self.sv_y[i]
        else:
            self.w = None
        self.is_fit = True
