import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional


class NeighboursGenerator:
    """Manages the process of obtaining the Neighbours for a given row"""

    def __init__(
        self,
        training_features: Union[np.ndarray, pd.DataFrame],
        data_row: np.ndarray,
        categorical_features: Optional[List[int]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Init function.

        Args:
            training_features (Union[np.ndarray, pd.DataFrame]): data used to train the bb model.
            data_row (np.ndarray): data point to be explained of shape (1 x features).
            categorical_features (Optional[List[int]]): list of integeter indicating the categorical features.
            random_state (Optional[int]): number to be used for random seeds.
        Returns:

            None.
        """
        if isinstance(training_features, pd.DataFrame):
            self.training_features = training_features.to_numpy()
        elif isinstance(training_features, list):
            self.training_features = np.array(training_features)
        else:
            self.training_features = training_features

        self.data_row = data_row
        self.total_features = self.training_features.shape[1]

        if categorical_features is None:
            self.cat_features = []
        else:
            self.cat_features = categorical_features

        self.cont_features = [
            i for i in range(self.total_features) if i not in self.cat_features
        ]
        self.total_cat_features = len(self.cat_features)
        self.total_cont_features = len(self.cont_features)

        if len(self.cat_features) == 0:
            self.type_features = "continuous"
        elif len(self.cont_features) == 0:
            self.type_features = "categorical"
        else:
            self.type_features = "mixed"

        self.random_state = random_state

    @staticmethod
    def to_dict(keys, values):
        return {key: value for key, value in zip(keys, values)}

    def estimate_distribution_categorical_features(self) -> Dict:
        """Estimates the distribution for each categorical variable.

        Args:
            None.

        Returns:
            distribution (dict): a dictionary containing the distribution for each categorical variable.

        """
        distribution = {}
        total = self.training_features.shape[0]
        if self.type_features in ["mixed", "categorical"]:
            for idx_feature in self.cat_features:
                feautre_values = self.training_features[:, idx_feature]
                unique, count = np.unique(feautre_values, return_counts=True)
                distribution[idx_feature] = self.to_dict(unique, count / total)
        return distribution

    def generate_cont_neighbours(self, num_samples: int) -> np.ndarray:
        """Generates a neighborhood around a prediction for continuous features.

        Args:
            num_samples (int): number of neighbours to generate.

        Returns:
            data (np.ndarray): original data point and neighbours with shape (num_samples x features).
        """
        # Get continuous features
        training_features_cont = self.training_features[:, self.cont_features]

        # Estimate the variance for continuous features
        sd_value = np.nanstd(training_features_cont, axis=0, dtype=np.float32)

        # Generate neighbours
        neighbours = self.random_state.normal(
            0, 1, size=(num_samples, self.total_cont_features)
        )
        data_row_cont = self.data_row[0][self.cont_features].astype(neighbours.dtype)
        neighbours *= sd_value
        neighbours += data_row_cont

        return neighbours

    def generate_cat_neighbours(self, num_samples: int) -> np.ndarray:
        """Generates a neighborhood around a prediction for continuous features.

        Args:
            num_samples (int): number of neighbours to generate.

        Returns:
            data (np.ndarray): original data point and neighbours with shape (num_samples x features).
        """
        # Generate distribution for categorical features
        cat_distribution = self.estimate_distribution_categorical_features()
        neighbours_list = []

        # Generate neighbours
        i = 0
        for _, feat_distribution in cat_distribution.items():
            feat_values = list(feat_distribution.keys())
            probability_values = list(feat_distribution.values())
            sample = self.random_state.choice(
                feat_values, size=num_samples, p=probability_values
            )
            neighbours_list.append(sample)
            i += 1

        neighbours = np.array(neighbours_list)
        neighbours = neighbours.T
        return neighbours

    def generate_neighbours(self, num_samples: int) -> np.ndarray:
        """Generates a neighborhood around a prediction.

        Args:
            num_samples (int): number of neighbours to generate.

        Returns:
            data (np.ndarray): original data point and neighbours with shape (num_samples x features).
        """
        # Generate neighbours for continuous features
        if self.type_features in ["continuous", "mixed"]:
            X_neigh_cont = self.generate_cont_neighbours(num_samples=num_samples)

        # Generate neighbours for categorical features
        if self.type_features in ["categorical", "mixed"]:
            X_neigh_cat = self.generate_cat_neighbours(num_samples=num_samples)

        # Create neighbours
        if self.type_features == "continuous":
            neighbours = np.copy(X_neigh_cont)
        elif self.type_features == "categorical":
            neighbours = np.copy(X_neigh_cat)
        else:
            neighbours = np.concatenate(
                (X_neigh_cont, X_neigh_cat), axis=1, dtype=object
            )

        return neighbours
