import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_circles
from typing import Optional


def linear_data_generator(n_samples: int,
                          n_features: int,
                          random_state: Optional[int] = None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X, y = make_blobs(n_samples, n_features, centers=2, cluster_std=2.0, random_state=random_state)
    # Assign '-1' instead of '0'
    is_zero = y < 1
    y[is_zero] = -1
    return train_test_split(X, y, random_state=random_state)


def non_linear_data_generator(n_samples: int,
                              random_state: Optional[int] = None) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    X, y = make_circles(n_samples, noise=0.1, factor=0.4, random_state=random_state)
    # Assign '-1' instead of '0'
    is_zero = y < 1
    y[is_zero] = -1
    return train_test_split(X, y, random_state=random_state)
