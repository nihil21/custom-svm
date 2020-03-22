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

    def __init__(self, kernel: Optional[str] = 'rbf', gamma: Optional[float] = None):
        # Note: the number of binary SVM classifiers needed will be known only when the dataset labels will be given

        # self.SVMs: list of tuples, each one of 3 elements: (SVM_binary_classifier, 1st_class_label, 2nd_class_label)
        self.SVMs = []
        # Gamma will be computed during fit process
        self.gamma = gamma
        # RBF used as kernel
        self.kernel = lambda x_i, x_j: np.exp(-self.gamma * np.inner(x_i - x_j))

    def fit(self, X: np.ndarray, y: np.ndarray):
        labels = np.unique(y)
        for i in range(0, labels.shape[0] - 1):
            for j in range(i + 1, labels.shape[0]):
                current_dataset_index = y == labels[i] or y == labels[j]
                current_X = X[current_dataset_index]
                current_y = y[current_dataset_index]
                svm = SVM(kernel=self.kernel, gamma=self.gamma)
                svm.fit(current_X, current_y)
                svm_tuple = (svm, labels[i], labels[j])
                self.SVMs.append(svm_tuple)
