import argparse
import numpy as np
from svm import SVM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', '--kernel', required=False, help="kernel type to use ['rbf'/'poly'/'sigmoid']")
    ap.add_argument('-g', '--gamma', required=False, help="gamma parameter of the kernel")
    args = vars(ap.parse_args())

    # Argument check
    kernel = None
    if args['kernel'] and (args['kernel'] in ('rbf', 'poly', 'sigmoid')):
        kernel = args['kernel']
    else:
        print("Kernel must be equal to either 'rbf', 'poly' or 'sigmoid'; 'rbf' will be used")
    gamma = None
    if args['gamma']:
        try:
            gamma = float(args['gamma'])
        except ValueError:
            print("Cannot convert {0:s} to floating point; default 'gamma' will be used".format(args['gamma']))

    n_samples = 30
    n_features = 3
    X = np.random.rand(n_samples, n_features)
    y = np.ones(shape=(n_samples, n_features))

    if kernel and gamma:
        svm = SVM(kernel, gamma)
    elif kernel and not gamma:
        svm = SVM(kernel=kernel)
    elif not kernel and gamma:
        svm = SVM(gamma=gamma)
    else:
        svm = SVM()
    svm.fit(X, y)


if __name__ == '__main__':
    main()
