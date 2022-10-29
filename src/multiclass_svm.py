import collections

import numpy as np

from svm import SVM


class MulticlassSVM:
    """Class implementing a Support Vector Machine for multi-classification purposes based on one-vs-one strategy.
    Given N different classes to classify, the algorithm provides N*(N-1)/2 SVM binary classifiers. Each classifier is
    trained to correctly classify 2 of the N given classes using in the training process only the entries in the
    dataset to which it corresponds a label of the 2 classes. Given an unseen example, the prediction of the class is
    computed deploying a voting schema among the classifiers.

    Parameters
    ----------
    kernel : {"linear", "rbf", "poly", "sigmoid"}
        Type of kernel function.
    gamma : float or None, default=None
        Value representing the gamma parameter of the kernel; if None, it will be computed automatically during fit.
    deg : int, default=3
        Value representing the degree of the "poly" kernel function.
    r : float, default=0.
        Value representing the r parameter of "poly" and "sigmoid" kernel functions.
    c : float or None, default=1
        Value regulating the trade-off between the amount of misclassified samples and the size of the margin
        (its "softness" decreases as C increases); if None, hard margin is employed (no tolerance towards
        misclassified samples).

    Attributes
    ----------
    _kernel : {"linear", "rbf", "poly", "sigmoid"}
        Type of kernel function.
    _gamma : float or None
        Value representing the gamma parameter of the kernel; if None, it will be computed automatically during fit.
    _deg : int
        Value representing the degree of the "poly" kernel function.
    _r : float
        Value representing the r parameter of "poly" and "sigmoid" kernel functions.
    _c : float or None
        Value regulating the trade-off between the amount of misclassified samples and the size of the margin
        (its "softness" decreases as C increases); if None, hard margin is employed (no tolerance towards
        misclassified samples).
    _svm_list : list of SVM
        List of triplets, each one comprising the SVM binary classifier, the label of the 1st class and the label of
        the 2nd class (1st class corresponds to sign "-", 2nd class corresponds to sign "+"); the number of binary
        SVM classifiers needed will be known only when the dataset labels are given.
    _labels : ndarray or None
        Integer labels.
    _support_vectors : set of tuple of (float, float)
        Set of support vectors.
    """

    def __init__(
            self,
            kernel: str = "linear",
            gamma: float | None = None,
            deg: int = 3,
            r: float = 0.,
            c: float | None = 1.
    ):
        # By default linear kernel is used
        self._kernel = kernel
        # If gamma is None, it will be computed during fit process
        self._gamma = gamma
        self._deg = deg
        self._r = r
        self._c = c
        self._svm_list = []
        self._labels = None
        self._support_vectors = set()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the SVM on the given training set.

        Parameters
        ----------
        x : ndarray
            Training data with shape (n_samples, n_features).
        y : ndarray
            Ground-truth labels.
        """
        # Check if labels are integers
        labels = np.unique(y)
        for label in labels:
            if not label.is_integer():
                raise ValueError(str(label) + " is not an integer value label")
        self._labels = np.array(labels, dtype=int)

        # Re-arrange training set per labels in a dictionary
        x_arranged_list = collections.defaultdict(list)
        for index, x_ in enumerate(x):
            x_arranged_list[y[index]].append(x_)

        # Convert to numpy array the previous dictionary
        x_arranged_numpy = {}
        for index in range(len(self._labels)):
            x_arranged_numpy[index] = np.array(x_arranged_list[index])

        for i in range(0, self._labels.shape[0] - 1):
            for j in range(i + 1, self._labels.shape[0]):
                current_x = np.concatenate((x_arranged_numpy[i], x_arranged_numpy[j]))
                current_y = np.concatenate((- np.ones((len(x_arranged_numpy[i]),), dtype=int),
                                           np.ones(len((x_arranged_numpy[j]),), dtype=int)))
                svm = SVM(kernel=self._kernel, gamma=self._gamma, deg=self._deg, r=self._r, c=self._c)
                svm.fit(current_x, current_y, verbosity=0)
                for sv in svm.sv_x:
                    self._support_vectors.add(tuple(sv.tolist()))
                svm_tuple = (svm, self._labels[i], self._labels[j])
                self._svm_list.append(svm_tuple)
        print('{0:d} support vectors found out of {1:d} data points'.format(len(self._support_vectors), len(x)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the class of the given data points. The voting process is based on the standard predict function
        for binary SVM classifiers, so the input entry is assigned to the class which wins the highest number of binary
        comparisons; to counteract the possible risk of draw, the predicted value before the application of the "sign"
        function in binary classifiers is stored as well.
        For each sample j, for each label i:
            - voting_schema[j][0][i] is the number of total comparisons won
            - voting_schema[j][1][i] is the cumulative sum of predicted values

        Parameters
        ----------
        x : ndarray
            Data points with shape (n_samples, n_features).

        Returns
        -------
        ndarray
            Results of the voting scheme.
        """
        voting_schema = np.zeros([len(x), 2, self._labels.shape[0]], dtype=float)
        for svm_tuple in self._svm_list:
            prediction = svm_tuple[0].project(x)
            for i in range(len(prediction)):
                if prediction[i] < 0:
                    voting_schema[i][0][svm_tuple[1]] += 1
                    voting_schema[i][1][svm_tuple[1]] += -1 * prediction[i]
                else:
                    voting_schema[i][0][svm_tuple[2]] += 1
                    voting_schema[i][1][svm_tuple[2]] += prediction[i]

        voting_results = np.zeros(len(voting_schema), dtype=int)
        for i in range(len(voting_schema)):
            sorted_votes = np.sort(voting_schema[i][0])
            # If the first two classes received a different number of votes there is no draw
            if sorted_votes[0] > sorted_votes[1]:
                voting_results[i] = voting_schema[i][0].argmax()
            # Otherwise return label of the class which has the highest cumulative sum of predicted values
            else:
                voting_results[i] = voting_schema[i][1].argmax()

        return voting_results
