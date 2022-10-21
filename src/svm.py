import itertools
import warnings

import cvxopt
import numpy as np
from matplotlib import pyplot as plt


class SVM:
    """Class implementing a Support Vector Machine: instead of minimising the primal function
        L_P(w, b, lambda_mat) = 1/2 ||w||^2 - sum_i{lambda_i[(w * x + b) - 1]},
    the dual function
        L_D(lambda_mat) = sum_i{lambda_i} - 1/2 sum_i{sum_j{lambda_i lambda_j y_i y_j K(x_i, x_j)}}
    is maximised.

    Parameters
    ----------
    kernel: str, default="linear"
        Type of kernel function ("linear", "rbf", "poly" or "sigmoid").
    gamma: float | None, default=None
        Value representing the gamma parameter of the kernel; if None, it will be computed automatically during fit.
    deg: int, default=3
        Value representing the degree of the "poly" kernel function.
    r: float, default=0.
        Value representing the r parameter of "poly" and "sigmoid" kernel functions.
    c: float | None, default=1.
        Value regulating the trade-off between the amount of misclassified samples and the size of the margin
        (its "softness" decreases as C increases); if None, hard margin is employed (no tolerance towards
        misclassified samples).

    Attributes
    ----------
    _kernel: str
        Type of kernel function ("linear", "rbf", "poly" or "sigmoid").
    _kernel_fn: Callable[[np.ndarray, np.ndarray], float]
        Kernel function.
    _gamma: float | None
        Value representing the gamma parameter of the kernel; if None, it will be computed automatically during fit.
    _lambdas: np.ndarray | None
        Lagrangian multipliers.
    _sv_x: np.ndarray | None
        Support vectors related to X.
    _sv_y: np.ndarray | None
        Support vectors related to y.
    _w: np.ndarray | None
        Matrix of hyperplane parameters.
    _b: float | None
        Hyperplane bias.
    _c: float | None
        Value regulating the trade-off between the amount of misclassified samples and the size of the margin
        (its "softness" decreases as C increases); if None, hard margin is employed (no tolerance towards
        misclassified samples).
    _is_fit: bool
        Whether the SVM is fit or not.
    """

    def __init__(
            self,
            kernel: str = "linear",
            gamma: float | None = None,
            deg: int = 3,
            r: float = 0.,
            c: float = 1.
    ):
        # Lagrangian's multipliers, hyperparameters and support vectors are initially set to None
        self._lambdas = None
        self._sv_x = None
        self._sv_y = None
        self._w = None
        self._b = None

        # If gamma is None, it will be computed during fit process
        self._gamma = gamma

        # Assign the right kernel
        self._kernel = kernel
        if kernel == "linear":
            self._kernel_fn = lambda x_i, x_j: np.dot(x_i, x_j)
        elif kernel == "rbf":
            self._kernel_fn = lambda x_i, x_j: np.exp(-self._gamma * np.dot(x_i - x_j, x_i - x_j))
        elif kernel == "poly":
            self._kernel_fn = lambda x_i, x_j: (self._gamma * np.dot(x_i, x_j) + r) ** deg
        elif kernel == "sigmoid":
            self._kernel_fn = lambda x_i, x_j: np.tanh(np.dot(x_i, x_j) + r)

        # Soft margin
        self._c = c

        self._is_fit = False

    def fit(self, x: np.ndarray, y: np.ndarray, verbosity: int = 1) -> None:
        """Fit the SVM on the given training set.

        Parameters
        ----------
        x: np.ndarray
            Training data with shape (n_samples, n_features).
        y: np.ndarray
            Ground-truth labels.
        verbosity: int, default=1
            Verbosity level in range [0, 3].
        """
        # If "verbosity" is outside range [0, 3], set it to default (1)
        if verbosity not in {0, 1, 2}:
            verbosity = 1

        n_samples, n_features = x.shape
        # If gamma was not specified in "__init__", it is set according to the "scale" approach
        if not self._gamma:
            self._gamma = 1 / (n_features * x.var())

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
        k = np.zeros(shape=(n_samples, n_samples))
        for i, j in itertools.product(range(n_samples), range(n_samples)):
            k[i, j] = self._kernel_fn(x[i], x[j])
        p = cvxopt.matrix(np.outer(y, y) * k)
        q = cvxopt.matrix(-np.ones(n_samples))
        # Compute G and h matrix according to the type of margin used
        if self._c:
            g = cvxopt.matrix(np.vstack((
                -np.eye(n_samples),
                np.eye(n_samples)
            )))
            h = cvxopt.matrix(np.hstack((
                np.zeros(n_samples),
                np.ones(n_samples) * self._c
            )))
        else:
            g = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        a = cvxopt.matrix(y.reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        # Set CVXOPT options
        cvxopt.solvers.options["show_progress"] = False
        cvxopt.solvers.options["maxiters"] = 200

        # Compute the solution using the quadratic solver
        try:
            sol = cvxopt.solvers.qp(p, q, g, h, a, b)
        except ValueError as e:
            print(f"Impossible to fit, try to change kernel parameters; CVXOPT raised Value Error: {e:s}")
            return
        # Extract Lagrange multipliers
        lambdas = np.ravel(sol["x"])
        # Find indices of the support vectors, which have non-zero Lagrange multipliers, and save the support vectors
        # as instance attributes
        is_sv = lambdas > 1e-5
        self._sv_x = x[is_sv]
        self._sv_y = y[is_sv]
        self._lambdas = lambdas[is_sv]
        # Compute b as 1/N_s sum_i{y_i - sum_sv{lambdas_sv * y_sv * K(x_sv, x_i}}
        sv_index = np.arange(len(lambdas))[is_sv]
        self._b = 0
        for i in range(len(self._lambdas)):
            self._b += self._sv_y[i]
            self._b -= np.sum(self._lambdas * self._sv_y * k[sv_index[i], is_sv])
        self._b /= len(self._lambdas)
        # Compute w only if the kernel is linear
        if self._kernel == "linear":
            self._w = np.zeros(n_features)
            for i in range(len(self._lambdas)):
                self._w += self._lambdas[i] * self._sv_x[i] * self._sv_y[i]
        else:
            self._w = None
        self._is_fit = True

        # Print results according to verbosity
        if verbosity in {1, 2}:
            print(f"{len(self._lambdas):d} support vectors found out of {n_samples:d} data points")
            if verbosity == 2:
                for i in range(len(self._lambdas)):
                    print(f"{i + 1:d}) X: {self._sv_x[i]}\ty: {self._sv_y[i]}\tlambda: {self._lambdas[i]:.2f}")
            print(f"Bias of the hyper-plane: {self._b:.3f}")
            print("Weights of the hyper-plane:", self._w)

    def project(
            self,
            x: np.ndarray,
            i: int | None = None,
            j: int | None = None
    ) -> np.ndarray:
        """Project data on the hyperplane.

        Parameters
        ----------
        x: np.ndarray
            Data points with shape (n_samples, n_features).
        i: int | None, default=None
            First dimension to plot (in the case of non-linear kernels).
        j: int | None, default=None
            Second dimension to plot (in the case of non-linear kernels).

        Returns
        -------
        proj: np.ndarray
            Projection of the points on the hyperplane.
        """
        # If the model is not fit, raise an exception
        if not self.is_fit:
            raise SVMNotFitError
        # If the kernel is linear and "w" is defined, the value of f(x) is determined by
        #   f(x) = X * w + b
        if self._w is not None:
            return np.dot(x, self._w) + self._b
        else:
            # Otherwise, it is determined by
            #   f(x) = sum_i{sum_sv{lambda_sv y_sv K(x_i, x_sv)}}
            y_predict = np.zeros(len(x))
            for k in range(len(x)):
                for lda, sv_x, sv_y in zip(self._lambdas, self._sv_x, self._sv_y):
                    # Extract the two dimensions from sv_x if "i" and "j" are specified
                    if i or j:
                        sv_x = np.array([sv_x[i], sv_x[j]])

                    y_predict[k] += lda * sv_y * self._kernel_fn(x[k], sv_x)
            return y_predict + self._b

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the given data points.

        Parameters
        ----------
        x: np.ndarray
            Data points with shape (n_samples, n_features).

        Returns
        -------
        label: np.ndarray
            Predicted labels.
        """
        # To predict the point label, only the sign of f(x) is considered
        return np.sign(self.project(x))

    def plot2d(
            self,
            x: np.ndarray,
            y: np.ndarray,
            x_min: float | None = None,
            x_max: float | None = None,
            y_min: float | None = None,
            y_max: float | None = None
    ) -> None:
        """Display the plots of data points color coded by predicted labels and margins.

        Parameters
        ----------
        x: np.ndarray
            Data points with shape (n_samples, n_features).
        y: np.ndarray
            Ground-truth labels.
        x_min: float | None, default=None
            Maximum x coordinate for the plot.
        x_max: float | None, default=None
            Minimum x coordinate for the plot.
        y_min: float | None, default=None
            Maximum y coordinate for the plot.
        y_max: float | None, default=None
            Minimum y coordinate for the plot.
        """
        # Get indexes of positive and negative labels
        is_pos = y > 0
        is_neg = y < 0

        # Get number of pairs to plot
        n_samples, n_features = x.shape
        pair_plots = list(itertools.combinations(np.arange(start=0, stop=n_features, step=1, dtype=int), 2))
        num_plots = len(pair_plots)
        if num_plots % 2 != 0:
            num_plots += 1

        # Check number of features
        if n_features > 2:
            # If the number of features is higher than 2, initialize a grid of subplots
            fig, ax = plt.subplots(nrows=num_plots // 2, ncols=2, figsize=(15, 30))
        elif n_features == 2:
            # If the number of features is 2, draw a single plot and wrap it in a fake grid
            fig, ax = plt.subplots(figsize=(15, 10))
            ax = np.array([
                [ax, 0],
                [0, 0]
            ])
        else:
            # Otherwise, return
            print("Number of dimensions must be 2 or higher.")
            return

        # Initialize plot counters
        p_i, p_j = 0, 0

        # Iterate over dimensions
        for i, j in pair_plots:
            # If bounds are not specified, compute them from maximum and minimum data
            if not x_min:
                x_min = np.min(x[:, i]) - 1
            if not x_max:
                x_max = np.max(x[:, i]) + 1
            if not y_min:
                y_min = np.min(x[:, j]) - 1
            if not y_max:
                y_max = np.max(x[:, j]) + 1

            # Initialize subplot
            ax[p_i, p_j].title.set_text(f"Dimensions {i:d}, {j:d}")
            ax[p_i, p_j].grid(True, which="both")
            ax[p_i, p_j].axhline(y=0, color="k")
            ax[p_i, p_j].axvline(x=0, color="k")

            # Plot training set points
            ax[p_i, p_j].plot(x[is_pos, i], x[is_pos, j], "bo")
            ax[p_i, p_j].plot(x[is_neg, i], x[is_neg, j], "ro")
            # Plot support vectors
            ax[p_i, p_j].scatter(self._sv_x[:, i], self._sv_x[:, j], s=100, c="g")

            # If the kernel is linear and "w" is defined, the hyperplane can be plotted using "w" and "b"
            if self._w is not None:
                # Function representing the hyperplane
                def f(x_: np.ndarray, w_0: float, w_1: float, b: float, c: float = 0):
                    return -(w_0 * x_ + b - c)/w_1

                # Plot the hyperplane
                x_s = np.linspace(x_min, x_max)
                ax[p_i, p_j].plot(x_s, f(x_s, self._w[i], self._w[j], self._b), "k")
                ax[p_i, p_j].plot(x_s, f(x_s, self._w[i], self._w[j], self._b, -1), "k--")
                ax[p_i, p_j].plot(x_s, f(x_s, self._w[i], self._w[j], self._b, 1), "k--")

            else:
                # Plot the contours of the decision function
                x1, x2 = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
                xs = np.array([[x1_, x2_] for x1_, x2_ in zip(np.ravel(x1), np.ravel(x2))])
                z = self.project(xs, i, j).reshape(x1.shape)

                # Suppress warnings and reactivate them after plotting contour
                warnings.filterwarnings("ignore")
                ax[p_i, p_j].contour(x1, x2, z, [0.], colors="k", linewidths=1, origin="lower")
                ax[p_i, p_j].contour(x1, x2, z + 1, [0.], colors="grey", linewidths=1, origin="lower")
                ax[p_i, p_j].contour(x1, x2, z - 1, [0.], colors="grey", linewidths=1, origin="lower")
                warnings.filterwarnings("default")

            ax[p_i, p_j].set(xlim=(x_min, x_max), ylim=(y_min, y_max))

            # Increment subplot counters
            if p_j == 0:
                p_j += 1
            else:
                p_i += 1
                p_j -= 1
        plt.show()

    @property
    def is_fit(self) -> bool:
        return self._is_fit

    @property
    def sv_x(self) -> np.ndarray:
        return self._sv_x

    @property
    def sv_y(self) -> np.ndarray:
        return self._sv_y


class SVMNotFitError(Exception):
    """Exception raised when the "project" or the "predict" method of an SVM object is called without fitting
    the model beforehand."""
