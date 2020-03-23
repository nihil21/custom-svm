import argparse
from svm import SVM
from data.sample_data_generator import *
from sklearn.metrics import accuracy_score

RND = 42
N_SAMP = 200
N_FEAT = 2


def fit_and_predict(svm: SVM,
                    X_train: np.ndarray,
                    X_test: np.ndarray,
                    y_train: np.ndarray,
                    y_test: np.ndarray):
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print('Accuracy: {0:.2f}%'.format(accuracy_score(y_test, y_pred) * 100))
    svm.plot2D(np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-g', '--gamma', required=False, help="'gamma' parameter of the kernel")
    ap.add_argument('-d', '--deg', required=False, help="'deg' parameter of the kernel")
    ap.add_argument('-r', '--r', required=False, help="'r' parameter of the kernel")
    args = vars(ap.parse_args())

    # Argument check
    svm_params = dict()
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

    print('-' * 60)
    print('\tLinear case')
    print('=' * 60)
    X_train, X_test, y_train, y_test = linear_data_generator(n_samples=N_SAMP,
                                                             n_features=N_FEAT,
                                                             random_state=RND)
    svm = SVM()
    fit_and_predict(svm, X_train, X_test, y_train, y_test)
    print('-' * 60)

    print('\tNon-linear case')
    print('=' * 60)
    svm_params['kernel'] = 'rbf'
    X_train, X_test, y_train, y_test = non_linear_data_generator(n_samples=N_SAMP,
                                                                 random_state=RND)
    svm = SVM(**svm_params)
    fit_and_predict(svm, X_train, X_test, y_train, y_test)
    print('-' * 60)


if __name__ == '__main__':
    main()
