import numpy as np
import itertools
import cvxopt
from matplotlib import pyplot as plt
from typing import Optional
import warnings


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
        C --- non-negative float regulating the trade-off between the amount of misclassified samples and
        the size of the margin (its 'softness' decreases as C increases); if it is set to 'None',
        hard margin is employed (no tolerance towards misclassified samples)
        is_fit --- boolean variable indicating whether the SVM is fit or not"""

    def __init__(self,
                 kernel: Optional[str] = 'linear',
                 gamma: Optional[float] = None,
                 deg: Optional[int] = 3,
                 r: Optional[float] = 0.,
                 C: Optional[float] = 1.):
        """Initializes the SVM object by setting the kernel function, its parameters and the soft margin;
        moreover, it sets to None the matrices of lagrangian multipliers and support vectors.
            :param kernel: string representing the kernel type ('linear'/'rbf'/'poly'/'sigmoid'); by default it is
            set to 'linear'
            :param gamma: optional floating point representing the gamma parameters of the kernel;
            by default it is 'None', since it will be computed automatically during fit
            :param deg: optional integer representing the degree of the 'poly' kernel function
            :param r: optional floating point representing the r parameter of 'poly' and 'sigmoid' kernel functions
            :param C: non-negative float regulating the trade-off between the amount of misclassified samples and
            the size of the margin"""
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
            self.kernel_fn = lambda x_i, x_j: np.exp(-self.gamma * np.dot(x_i - x_j, x_i - x_j))
        elif kernel == 'poly':
            self.kernel_fn = lambda x_i, x_j: (self.gamma * np.dot(x_i, x_j) + r) ** deg
        elif kernel == 'sigmoid':
            self.kernel_fn = lambda x_i, x_j: np.tanh(np.dot(x_i, x_j) + r)

        # Soft margin
        self.C = C

        self.is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray, verbosity: Optional[int] = 1):
        # If 'verbosity' is outside range (0-3), set it to default (1)
        if verbosity not in {0, 1, 2}:
            verbosity = 1

        n_samples, n_features = X.shape
        # If gamma was not specified in '__init__', it is set according to the 'scale' approach
        if not self.gamma:
            self.gamma = 1/(n_features * X.var())

        # max{L_D(Lambda)} can be rewritten as
        #   min{1/2 Lambda^T H Lambda - 1^T Lambda}
        #       s.t. -lambda_i <= 0
        #       s.t. lambda_i <= c
        #       s.t. y^t Lambda = 0
        # where H[i, j] = y_i y_j K(x_i, x_j)
        # This form is conform to the signature of the quadratic solver provided by CVXOPT library:
        #   min{1/2 x^T P x + q^T x}
        #       s.t. G x <= h
        #       s.t. A x = b
        K = np.zeros(shape=(n_samples, n_samples))
        for i, j in itertools.product(range(n_samples), range(n_samples)):
            K[i, j] = self.kernel_fn(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        # Compute G and h matrix according to the type of margin used
        if self.C:
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples),
                                         np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples),
                                         np.ones(n_samples) * self.C)))
        else:
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        A = cvxopt.matrix(y.reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        # Set CVXOPT options
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['maxiters'] = 200

        # Compute the solution using the quadratic solver
        try:
            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        except ValueError as e:
            print('Impossible to fit, try to change kernel parameters; CVXOPT raised Value Error: {0:s}'.format(e))
            return
        # Extract Lagrange multipliers
        lambdas = np.ravel(sol['x'])
        # Find indices of the support vectors, which have non-zero Lagrange multipliers, and save the support vectors
        # as instance attributes
        is_sv = lambdas > 1e-5
        self.sv_X = X[is_sv]
        self.sv_y = y[is_sv]
        self.lambdas = lambdas[is_sv]
        # Compute b as 1/N_s sum_i{y_i - sum_sv{lambdas_sv * y_sv * K(x_sv, x_i}}
        sv_index = np.arange(len(lambdas))[is_sv]
        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lambdas * self.sv_y * K[sv_index[i], is_sv])
        self.b /= len(self.lambdas)
        # Compute w only if the kernel is linear
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for i in range(len(self.lambdas)):
                self.w += self.lambdas[i] * self.sv_X[i] * self.sv_y[i]
        else:
            self.w = None
        self.is_fit = True

        # Print results according to verbosity
        if verbosity in {1, 2}:
            print('{0:d} support vectors found out of {1:d} data points'.format(len(self.lambdas), n_samples))
            if verbosity == 2:
                for i in range(len(self.lambdas)):
                    print('{0:d}) X: {1}\ty: {2}'.format(i + 1, self.sv_X[i], self.sv_y[i]))
                    print('Lagrangian multipliers:', self.lambdas)
            print('Bias of the hyper-plane: {0:.3f}'.format(self.b))
            print('Weights of the hyper-plane:', self.w)

    def project(self,
                X: np.ndarray,
                i: Optional[int] = None,
                j: Optional[int] = None):
        # If the model is not fit, raise an exception
        if not self.is_fit:
            raise SVMNotFitError
        # If the kernel is linear and 'w' is defined, the value of f(x) is determined by
        #   f(x) = X * w + b
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            # Otherwise, it is determined by
            #   f(x) = sum_i{sum_sv{lambda_sv y_sv K(x_i, x_sv)}}
            y_predict = np.zeros(len(X))
            for k in range(len(X)):
                for lda, sv_X, sv_y in zip(self.lambdas, self.sv_X, self.sv_y):
                    # Extract the two dimensions from sv_X if 'i' and 'j' are specified
                    if i or j:
                        sv_X = np.array([sv_X[i], sv_X[j]])

                    y_predict[k] += lda * sv_y * self.kernel_fn(X[k], sv_X)
            return y_predict + self.b

    def predict(self, X: np.ndarray) -> int:
        # To predict the point label, only the sign of f(x) is considered
        return np.sign(self.project(X))

    def plot2D(self,
               X: np.ndarray,
               y: np.ndarray,
               x_min: Optional[float] = None,
               x_max: Optional[float] = None,
               y_min: Optional[float] = None,
               y_max: Optional[float] = None):
        # Get indexes of positive and negative labels
        is_pos = y > 0
        is_neg = y < 0

        # Get number of pairs to plot
        n_samples, n_features = X.shape
        pair_plots = list(itertools.combinations(np.arange(start=0, stop=n_features, step=1, dtype=np.int), 2))
        num_plots = len(pair_plots)
        if num_plots % 2 != 0:
            num_plots += 1

        # Check number of features
        if n_features > 2:
            # If the number of features is higher than 2, initialize a grid of subplots
            fig, ax = plt.subplots(nrows=int(num_plots/2), ncols=2, figsize=(15, 30))
        elif n_features == 2:
            # If the number of features is 2, draw a single plot and wrap it in a fake grid
            fig, ax = plt.subplots(figsize=(15, 10))
            ax = np.array([[ax, 0],
                           [0, 0]])
        else:
            # Otherwise, return
            print('Number of dimensions must be 2 or higher.')
            return

        # Initialize plot counters
        p_i, p_j = 0, 0

        # Iterate over dimensions
        for i, j in pair_plots:
            # If bounds are not specified, compute them from maximum and minimum data
            if not x_min:
                x_min = np.min(X[:, i]) - 1
            if not x_max:
                x_max = np.max(X[:, i]) + 1
            if not y_min:
                y_min = np.min(X[:, j]) - 1
            if not y_max:
                y_max = np.max(X[:, j]) + 1

            # Initialize subplot
            ax[p_i, p_j].title.set_text('Dimensions {0:d}, {1:d}'.format(i, j))
            ax[p_i, p_j].grid(True, which='both')
            ax[p_i, p_j].axhline(y=0, color='k')
            ax[p_i, p_j].axvline(x=0, color='k')

            # Plot training set points
            ax[p_i, p_j].plot(X[is_pos, i], X[is_pos, j], 'bo')
            ax[p_i, p_j].plot(X[is_neg, i], X[is_neg, j], 'ro')
            # Plot support vectors
            ax[p_i, p_j].scatter(self.sv_X[:, i], self.sv_X[:, j], s=100, c='g')

            # If the kernel is linear and 'w' is defined, the hyperplane can be plotted using 'w' and 'b'
            if self.w is not None:
                # Function representing the hyperplane
                def f(x: np.ndarray, w_0: float, w_1: float, b: float, c: Optional[float] = 0):
                    return -(w_0 * x + b - c)/w_1

                # Plot the hyperplane
                x_s = np.linspace(x_min, x_max)
                ax[p_i, p_j].plot(x_s, f(x_s, self.w[i], self.w[j], self.b), 'k')
                ax[p_i, p_j].plot(x_s, f(x_s, self.w[i], self.w[j], self.b, -1), 'k--')
                ax[p_i, p_j].plot(x_s, f(x_s, self.w[i], self.w[j], self.b, 1), 'k--')

            else:
                # Plot the contours of the decision function
                X1, X2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
                Xs = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
                Z = self.project(Xs, i, j).reshape(X1.shape)

                # Suppress warnings and reactivate them after plotting contour
                warnings.filterwarnings('ignore')
                ax[p_i, p_j].contour(X1, X2, Z, [0.], colors='k', linewidths=1, origin='lower')
                ax[p_i, p_j].contour(X1, X2, Z + 1, [0.], colors='grey', linewidths=1, origin='lower')
                ax[p_i, p_j].contour(X1, X2, Z - 1, [0.], colors='grey', linewidths=1, origin='lower')
                warnings.filterwarnings('default')

            ax[p_i, p_j].set(xlim=(x_min, x_max), ylim=(y_min, y_max))

            # Increment subplot counters
            if p_j == 0:
                p_j += 1
            else:
                p_i += 1
                p_j -= 1
        plt.show()


class SVMNotFitError(Exception):
    """Exception raised when the 'project' or the 'predict' method of an SVM object is called without fitting
    the model beforehand."""
