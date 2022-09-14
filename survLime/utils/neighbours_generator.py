import numpy as np
import pandas as pd
from typing import List, Dict, Union


class NeighboursGenerator:
    """Manages the process of obtaining the Neighbours for a given row"""

    def __init__(
        self,
        training_data: Union[np.ndarray, pd.DataFrame],
        data_row: np.ndarray,
        categorical_features: List[int] = None,
    ) -> None:

        """Init function.

        Args:
            training_data (Union[np.ndarray, pd.DataFrame]): data used to train the bb model
            data_row (np.ndarray): data point to be explained of shape (1 x features)
            categorical_features (List[int]): list of integeter indicating the categorical variables

        Returns:
            None
        """

        if isinstance(training_data, pd.DataFrame):
            self.training_data = training_data.to_numpy()
        else:
            self.training_data = training_data
        self.data_row = data_row
        self.total_features = training_data.shape[1]

        if categorical_features is None:
            self.cat_features = []
        else:
            self.cat_features = categorical_features

        self.cont_features = [
            i for i in range(self.total_features) if i not in categorical_features
        ]
        self.total_cat_features = len(self.cat_features)
        self.total_cont_features = len(self.cont_features)

    @staticmethod
    def to_dict(keys, values):
        return {key: value for key, value in zip(keys, values)}

    def estimate_distribution_categorical_features(self) -> Dict:
        """Estimates the distribution for each categorical variable

        Returns:
            distribution (dict): a dictionary containing the distribution for each categorical variable
        """

        distribution = {}
        if self.total_cat_features > 0:
            for idx_feature in self.cat_features:
                feautre_values = self.training_data[:, idx_feature]
                unique, count = np.unique(feautre_values, return_counts=True)
                distribution[idx_feature] = self.to_dict(unique, count)
        return distribution
