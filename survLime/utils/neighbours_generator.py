import numpy as np
from typing import List


class NeighboursGenerator:
    """Manages the process of obtaining the Neighbours for a given row"""

    def __init__(
        self,
        data: np.ndarray,
        data_row: np.ndarray,
        categorical_features: List[int] = None,
    ) -> None:
        """Init function.

        Args:
            data (np.ndarray): dataset from which to estimate the distribution of the variables
            data_row (np.ndarray): data point to be explained of shape (1 x features)
            categorical_features (List[int]): list of integeter indicating the categorical variables

        Returns:
            None
        """
        self.data = data
        self.data_row = data_row
        self.total_features = data.shape[1]
        self.categorical_features = categorical_features
        if categorical_features is None:
            self.total_categorical_features = 0
        else:
            self.categorical_features = data.shape[1] - len(categorical_features)
