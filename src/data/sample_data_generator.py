import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_circles
from typing import Optional


def linear_data_generator(
        n_samples: int,
        n_features: int,
        random_state: int | None = None
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    x, y = make_blobs(n_samples, n_features, centers=2, cluster_std=2.0, random_state=random_state)
    # Assign '-1' instead of '0'
    is_zero = y < 1
    y[is_zero] = -1
    return train_test_split(x, y, random_state=random_state)


def semi_linear_data_generator(
        n_samples: int,
        n_features: int,
        random_state: int | None = None
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    x, y = make_blobs(n_samples, n_features, centers=[[-2, 2], [2, -2]], cluster_std=0.7, random_state=random_state)
    # Assign '-1' instead of '0'
    x[-1] = np.array([0.5, - 0.7])
    x[-2] = np.array([0.7, - 0.6])
    y[-1] = 0
    y[-2] = 0
    is_zero = y < 1
    y[is_zero] = -1
    return train_test_split(x, y, random_state=random_state)


def non_linear_data_generator(
        n_samples: int,
        random_state: int | None = None
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    x, y = make_circles(n_samples, noise=0.05, factor=0.4, random_state=random_state)
    # Assign '-1' instead of '0'
    is_zero = y < 1
    y[is_zero] = -1
    return train_test_split(x, y, random_state=random_state)
