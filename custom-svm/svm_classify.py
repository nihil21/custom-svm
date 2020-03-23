import argparse
import numpy as np
from svm import SVM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-k', '--kernel', required=False, help="kernel type to use ['linear'/'rbf'/'poly'/'sigmoid']")
    ap.add_argument('-g', '--gamma', required=False, help="'gamma' parameter of the kernel")
    ap.add_argument('-d', '--deg', required=False, help="'deg' parameter of the kernel")
    ap.add_argument('-r', '--r', required=False, help="'r' parameter of the kernel")
    args = vars(ap.parse_args())

    # Argument check
    svm_params = dict()
    if args['kernel']:
        if args['kernel'] in ('linear', 'rbf', 'poly', 'sigmoid'):
            svm_params['kernel'] = args['kernel']
        else:
            print("Kernel must be equal to either 'linear', 'rbf', 'poly' or 'sigmoid'; 'linear' will be used.")
    else:
        print("'Linear' kernel will be used")
    if args['gamma']:
        try:
            svm_params['gamma'] = float(args['gamma'])
        except ValueError:
            print("Cannot convert {0:s} to floating point; default 'gamma' will be used".format(args['gamma']))
    if args['deg']:
        try:
            svm_params['deg'] = int(args['deg'])
        except ValueError:
            print("Cannot convert {0:s} to integer; default 'deg' will be used".format(args['deg']))
    if args['r']:
        try:
            svm_params['r'] = float(args['r'])
        except ValueError:
            print("Cannot convert {0:s} to floating point; default 'r' will be used".format(args['r']))

    n_samples = 30
    n_features = 3
    X = np.random.rand(n_samples, n_features)
    y = np.ones(shape=(n_samples, 1))

    svm = SVM(**svm_params)
    svm.fit(X, y)


if __name__ == '__main__':
    main()
