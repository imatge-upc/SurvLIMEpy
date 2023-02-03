import numpy as np
import pandas as pd
from typing import Union, Optional, Iterable


class NeighboursGenerator:
    """Manages the process of obtaining the Neighbours for a given row"""

    def __init__(
        self,
        training_features: Union[np.ndarray, pd.DataFrame],
        data_row: np.ndarray,
        sigma: float,
        random_state: Optional[int] = None,
    ) -> None:
        """Init function.

        Args:
            training_features (Union[np.ndarray, pd.DataFrame]): data used to train the bb model.
            data_row (np.ndarray): data point to be explained of shape (1 x features).
            sigma (float): standard deviation used to generate the neighbours.
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

        self.sigma = sigma
        self.data_row = data_row
        self.total_features = self.training_features.shape[1]
        self.random_state = random_state

    def generate_neighbours(self, num_samples: int) -> np.ndarray:
        """Generates a neighborhood around a prediction.

        Args:
            num_samples (int): number of neighbours to generate.

        Returns:
            data (np.ndarray): original data point and neighbours with shape (num_samples x features).
        """
        # Generate neighbours
        p = self.training_features.shape[1]
        sd_vector = np.std(
            self.training_features, axis=0, dtype=self.training_features.dtype
        )
        sd_matrix = self.sigma * np.diag(sd_vector)

        normal_standard = self.random_state.normal(
            loc=0,
            scale=1,
            size=(num_samples, p),
        )
        neighbours = np.matmul(normal_standard, sd_matrix) + self.data_row
        return neighbours
