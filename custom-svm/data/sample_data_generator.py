import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from typing import Optional


def linear_data_generator(n_samples: int,
                          n_features: int,
                          random_state: Optional[int] = None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    rnd_state = np.random.RandomState(seed=random_state)
    X = rnd_state.rand(n_samples, n_features)
    y = np.ones(n_samples, dtype=np.int)
    # Produce the linear dataset s.t. positives and negatives are separated by
    #   y = x - 0.1
    for i in range(n_samples):
        if (X[i, 0] - 0.1) > X[i, 1]:
            y[i] = -y[i]
    return train_test_split(X, y, random_state=random_state)


def non_linear_data_generator(n_samples: int,
                              random_state: Optional[int] = None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X, y = make_circles(n_samples, noise=0.05, factor=0.4, random_state=random_state)
    # Assign '-1' instead of '0'
    is_zero = y < 1
    y[is_zero] = -1
    return train_test_split(X, y, random_state=random_state)
