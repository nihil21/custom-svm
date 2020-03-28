import collections
from typing import Optional
import numpy as np
from svm import SVM


class MulticlassSVM:
    """Class implementing a Support Vector Machine for multi-classification purposes based on one-vs-one strategy.
        Given N different classes to classify, the algorithm provides N*(N-1)/2 SVM binary classifiers.
        Each classifier is trained to correctly classify 2 of the N given classes using in the training process
        only the entries in the dataset to which it corresponds a label of the 2 classes.
        Given an unseen example, the prediction of the class is computed deploying a voting schema among the classifiers
    """

    def __init__(self,
                 kernel: Optional[str] = 'linear',
                 gamma: Optional[float] = None,
                 deg: Optional[int] = 3,
                 r: Optional[float] = 0.0,
                 C: Optional[float] = 1.):
        # self.SVMs: list of tuples, each one of 3 elements: (SVM_binary_classifier, 1st_class_label, 2nd_class_label)
        #            1st_class_label corresponds to sign "-", 2nd_class_label to sign "+"
        # Note: the number of binary SVM classifiers needed will be known only when the dataset labels will be given
        self.SVMs = []
        # By default linear kernel is used
        self.kernel = kernel
        # If gamma is None, it will be computed during fit process
        self.gamma = gamma
        self.deg = deg
        self.r = r
        self.C = C
        self.labels = None
        self.support_vectors = set()

    def fit(self, X: np.ndarray, y: np.ndarray):
        # check if labels are integers
        labels = np.unique(y)
        for label in labels:
            if not label.is_integer():
                raise ValueError(str(label) + " is not an integer value label")
        self.labels = np.array(labels, dtype=int)

        # re-arrange training set per labels in a dictionary
        X_arranged_list = collections.defaultdict(list)
        for index, x in enumerate(X):
            X_arranged_list[y[index]].append(x)

        # convert to numpy array the previous dictionary
        X_arranged_numpy = {}
        for index in range(len(self.labels)):
            X_arranged_numpy[index] = np.array(X_arranged_list[index])

        for i in range(0, self.labels.shape[0] - 1):
            for j in range(i + 1, self.labels.shape[0]):
                current_X = np.concatenate((X_arranged_numpy[i], X_arranged_numpy[j]))
                current_y = np.concatenate((- np.ones((len(X_arranged_numpy[i]),), dtype=int),
                                           np.ones(len((X_arranged_numpy[j]),), dtype=int)))
                svm = SVM(kernel=self.kernel, gamma=self.gamma, deg=self.deg, r=self.r, C=self.C)
                svm.fit(current_X, current_y, verbosity=0)
                for sv in svm.sv_X:
                    self.support_vectors.add(tuple(sv.tolist()))
                svm_tuple = (svm, self.labels[i], self.labels[j])
                self.SVMs.append(svm_tuple)
        print('{0:d} support vectors found out of {1:d} data points'.format(len(self.support_vectors), len(X)))

    def predict(self, X: np.ndarray):
        """The voting process is based on the standard predict function for binary SVM classifiers, so the input entry
           is assigned to the class which wins the highest number of binary comparisons.
           Anyway, to counteract the possible risk of draw, the predicted value before the application of 'sign'
           function in binary classifiers is stored as well. These latter values are used to deal with draws.
           For each sample j, for each label i:
           - voting_schema[j][0][i] is the number of total comparisons won
           - voting_schema[j][1][i] is the cumulative sum of predicted values"""

        voting_schema = np.zeros([len(X), 2, self.labels.shape[0]], dtype=float)
        for svm_tuple in self.SVMs:
            prediction = svm_tuple[0].project(X)
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
            # if the first two classes received a different number of votes there is no draw
            if sorted_votes[0] > sorted_votes[1]:
                voting_results[i] = voting_schema[i][0].argmax()
            # otherwise return label of the class which has highest cumulative sum of predicted values
            else:
                voting_results[i] = voting_schema[i][1].argmax()

        return voting_results
