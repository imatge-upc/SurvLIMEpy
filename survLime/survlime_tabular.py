from functools import partial
import numpy as np
import cvxpy as cp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from sksurv.nonparametric import nelson_aalen_estimator
from typing import Callable, Tuple


class LimeTabularExplainer:
    """To DO: change explanation
    Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(
        self,
        training_data: np.ndarray,
        target_data: np.ndarray,
        H0: np.ndarray = None,
        kernel_width: float = None,
        kernel: Callable = None,
        sample_around_instance: bool = False,
        random_state: int = None,
    ) -> None:

        """Init function.

        Args:
            To do

        Returns:
            To do

        """

        self.random_state = check_random_state(random_state)
        self.sample_around_instance = sample_around_instance
        self.train_events = [y[0] for y in target_data]
        self.train_times = [y[1] for y in target_data]
        if H0 is None:
            self.H0 = self.compute_nelson_aalen_estimator(
                self.train_events, self.train_times
            )
        else:
            self.H0 = H0

        # Validate H0 has the correct format
        # self.validate_H0(self.H0)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * 0.75
        kernel_width = float(kernel_width)

        if kernel is None:

            def kernel(d: np.ndarray, kernel_width: float) -> np.ndarray:
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.kernel_fn = partial(kernel, kernel_width=kernel_width)

        # Though set has no role to play if training data stats are provided
        # TO DO - Show Cris!
        # Instantiate an Scalar that will become important
        # take notice in the argument with_mean = False
        # We won't scale the data with the .transform method anyway
        # I tried switching it to false and it gave the same mean and variance
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)

    @staticmethod
    def compute_nelson_aalen_estimator(
        event: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        nelson_aalen = nelson_aalen_estimator(event, time)
        H0 = nelson_aalen[1]
        m = H0.shape[0]
        H0 = np.reshape(H0, newshape=(m, 1))
        return H0

    @staticmethod
    def validate_H0(H0: np.ndarray) -> None:
        if len(H0.shape) != 2:
            raise IndexError('H0 must be a 2 dimensional array.')
        if H0.shape[1] != 1:
            raise IndexError('The length of the last axis of must be equal to 1.')

    def explain_instance(
        self,
        data_row: np.ndarray,
        predict_fn: Callable,
        num_samples: int = 5000,
        distance_metric: str = 'euclidean',
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Generates explanations for a prediction.

        To do

        Args:
            To do

        Returns:
            To do
        """

        scaled_data = self.generate_neighbours(data_row, num_samples)
        distances = sklearn.metrics.pairwise_distances(
            scaled_data, scaled_data[0].reshape(1, -1), metric=distance_metric  # TO DO
        ).ravel()

        weights = self.kernel_fn(distances)

        # Solution for the optimisation problems
        return self.solve_opt_problem(
            predict_fn=predict_fn,
            num_samples=num_samples,
            weights=weights,
            H0=self.H0,
            scaled_data=scaled_data,
            verbose=verbose,
        )

    def solve_opt_problem(
        self,
        predict_fn: Callable,
        num_samples: int,
        weights: np.ndarray,
        H0: np.ndarray,
        scaled_data: np.ndarray,
        verbose: float,
    ) -> Tuple[np.ndarray, float]:
        """Solves the convex problem proposed in: https://arxiv.org/pdf/2003.08371.pdfF

        Args:
            # To do

        Returns:
           To do
        """
        epsilon = 0.00000001
        num_features = scaled_data.shape[1]
        m = len(set(self.train_times))
        # To do: validate H_i_j_wc
        H_i_j_wc = predict_fn(scaled_data)
        times_to_fill = list(set(self.train_times))
        times_to_fill.sort()
        log_correction = np.divide(H_i_j_wc, np.log(H_i_j_wc + epsilon))

        # Varible to look for
        b = cp.Variable((num_features, 1))

        # Reshape and log of predictions
        H = np.reshape(np.array(H_i_j_wc), newshape=(num_samples, m))
        LnH = np.log(H)

        # Log of baseline cumulative hazard
        LnH0 = np.log(H0)

        # Compute the log correction
        logs = np.reshape(log_correction, newshape=(num_samples, m))

        # Distance weights
        w = np.reshape(weights, newshape=(num_samples, 1))

        # Time differences
        t = self.train_times.copy()
        t.append(t[-1] + epsilon)
        t.sort()
        delta_t = [t[i + 1] - t[i] for i in range(m)]
        delta_t = np.reshape(np.array(delta_t), newshape=(m, 1))

        # Matrices to produce the proper sizes
        ones_N = np.ones(shape=(num_samples, 1))
        ones_m_1 = np.ones(shape=(m, 1))
        B = np.dot(ones_N, LnH0.T)
        C = LnH - B
        Z = scaled_data @ b
        D = Z @ ones_m_1.T
        E = C - D
        E_sq = cp.square(E)
        V_sq = cp.square(logs)
        F = cp.multiply(E_sq, V_sq)
        G = F @ delta_t
        funct = G.T @ w
        objective = cp.Minimize(funct)
        prob = cp.Problem(objective)
        result = prob.solve(verbose=verbose)
        return b.value, result  # H_i_j_wc, weights, log_correction, scaled_data,

    def generate_neighbours(self, data_row: np.ndarray, num_samples: int) -> np.ndarray:
        """Generates a neighborhood around a prediction.

        To do

        Args:
            To do

        Returns:
            To do
        """
        num_cols = data_row.shape[0]
        data = np.zeros((num_samples, num_cols))
        instance_sample = data_row
        scale = self.scaler.scale_
        mean = self.scaler.mean_
        data = self.random_state.normal(0, 1, num_samples * num_cols).reshape(
            num_samples, num_cols
        )
        if self.sample_around_instance:
            data = data * scale + instance_sample
        else:
            data = data * scale + mean
        data[0] = data_row.copy()
        return data
