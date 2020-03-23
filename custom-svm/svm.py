import numpy as np
import itertools
import cvxopt
from matplotlib import pyplot as plt
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
            self.kernel_fn = lambda x_i, x_j: np.dot(x_i, x_j)
        elif kernel == 'rbf':
            self.kernel_fn = lambda x_i, x_j: np.exp(-self.gamma * np.inner(x_i - x_j, x_i - x_j))
        elif kernel == 'poly':
            self.kernel_fn = lambda x_i, x_j: (gamma * np.dot(x_i, x_j) + r) ** deg
        elif kernel == 'sigmoid':
            self.kernel_fn = lambda x_i, x_j: np.tanh(np.dot(x_i, x_j) + r)

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
        is_sv = lambdas > 1e-5
        self.sv_X = X[is_sv]
        self.sv_y = y[is_sv]
        self.lambdas = lambdas[is_sv]
        print('{0:d} support vectors found out of {1:d} data points:'.format(len(self.lambdas), n_samples))
        for i in range(len(self.lambdas)):
            print('{0:d}) X: {1}\ty: {2}'.format(i+1, self.sv_X[i], self.sv_y[i]))

        # Compute b as 1/N_s sum_i{y_i - sum_sv{lambdas_sv * y_sv * K(x_sv, x_i}}
        sv_index = np.arange(len(lambdas))[is_sv]
        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lambdas * self.sv_y * K[sv_index[i], is_sv])
        self.b /= len(self.lambdas)
        print('Bias of the hyper-plane: {0:f}'.format(self.b))
        # Compute w only if the kernel is linear
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for i in range(len(self.lambdas)):
                self.w += self.lambdas[i] * self.sv_X[i] * self.sv_y[i]
            print('Weights of the hyper-plane:')
            print(self.w)
        else:
            self.w = None
        self.is_fit = True

    def project(self, X: np.ndarray) -> float:
        # If the kernel is linear and 'w' is defined, the value of f(x) is determined by
        #   f(x) = X * w + b
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            # Otherwise, it is determined by
            #   f(x) = sum_i{sum_sv{lambda_sv y_sv K(x_i, x_sv)}}
            y_predict = np.zeros(len(X))
            for x, y in zip(X, y_predict):
                for lda, sv_X, sv_y in zip(self.lambdas, self.sv_X, self.sv_y):
                    y += lda * sv_y * self.kernel_fn(x, sv_X)
            return y_predict + self.b

    def predict(self, X: np.ndarray) -> int:
        # To predict the point label, only the sign of f(x) is considered
        return np.sign(self.project(X))

    def plot2D(self, X: np.ndarray, y: np.ndarray):
        # If the dimension of the data is greater than 2, return
        if X.shape[1] > 2:
            print('Cannot plot data, dimension is greater than 2')
            return
        # If the kernel is linear and 'w' is defined, the hyperplane can be plotted using 'w' and 'b'
        if self.w is not None:
            # Function representing the hyperplane
            def f(x: np.ndarray, w: np.ndarray, b: float, c: Optional[float] = 0):
                return -(w[0] * x + b - c)/w[1]
            is_pos = y > 0
            is_neg = y < 0
            x_s = np.linspace(np.min(X), np.max(X))
            plt.plot(x_s, f(x_s, self.w, self.b), 'k')
            plt.plot(x_s, f(x_s, self.w, self.b, -1), 'k--')
            plt.plot(x_s, f(x_s, self.w, self.b, 1), 'k--')
            plt.scatter(X[is_pos, 0], X[is_pos, 1], c='b')
            plt.scatter(X[is_neg, 0], X[is_neg, 1], c='r')
            plt.show()
