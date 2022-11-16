from functools import partial
from typing import Callable, Tuple, Union, List
import numpy as np
import cvxpy as cp
import sklearn
import sklearn.preprocessing
import pandas as pd
from sklearn.utils import check_random_state
from sksurv.nonparametric import nelson_aalen_estimator
from survlime.utils.optimization import OptFuncionMaker
from survlime.utils.neighbours_generator import NeighboursGenerator


class SurvLimeExplainer:
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
        training_data: Union[np.ndarray, pd.DataFrame],
        target_data: Union[np.ndarray, pd.DataFrame],
        model_output_times: np.ndarray,
        categorical_features: List[int] = None,
        H0: np.ndarray = None,
        kernel_width: float = None,
        kernel: Callable = None,
        sample_around_instance: bool = False,
        random_state: int = None,
    ) -> None:

        """Init function.

        Args:
            training_data (Union[np.ndarray, pd.DataFrame]): data used to train the bb model
            target_data (Union[np.ndarray, pd.DataFrame]): target data used to train the bb model
            categorical_features (List[int]): list of integeter indicating the categorical features
            model_output_times (np.ndarray): output times of the bb model
            H0 (np.ndarray): baseline cumulative hazard
            kernel_width (float): width of the kernel to be used for computing distances
            kernel (Callable): kernel function to be used for computing distances
            sample_around_instance (bool): whether we sample around instances or not
            random_state (int): number to be used for random seeds

        Returns:
            None
        """

        self.random_state = check_random_state(random_state)
        self.sample_around_instance = sample_around_instance
        self.training_data = training_data
        self.train_events = [y[0] for y in target_data]
        self.train_times = [y[1] for y in target_data]
        self.categorical_features = categorical_features
        self.model_output_times = model_output_times
        if H0 is None:
            self.H0 = self.compute_nelson_aalen_estimator(
                self.train_events, self.train_times
            )
        else:
            self.H0 = H0

        # Validate H0 has the correct format
        # self.validate_H0(self.H0)

        if kernel_width is None:
            kernel_width = np.sqrt(self.training_data.shape[1]) * 0.75
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
        self.scaler.fit(self.training_data)

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
            raise IndexError("H0 must be a 2 dimensional array.")
        if H0.shape[1] != 1:
            raise IndexError("The length of the last axis of must be equal to 1.")

    def explain_instance(
        self,
        data_row: np.ndarray,
        predict_fn: Callable,
        num_samples: int = 5000,
        distance_metric: str = "euclidean",
        norm: Union[float, str] = 2,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Generates explanations for a prediction.

        Args:
            data_row (np.ndarray): data point to be explained
            predict_fn (Callable): function that computes cumulative hazard
            num_samples (int): number of neighbours to use
            distance_metric (str): metric to be used for computing neighbours distance to the original point
            norm (Union[float, str]): number
            verbose (bool = False):

        Returns:
            b.values (np.ndarray): obtained weights from the convex problem.
            result (float): residual value of the convex problem.
        """

        neighbours_generator = NeighboursGenerator(
            training_data=self.training_data,
            data_row=data_row,
            categorical_features=self.categorical_features,
            random_state=self.random_state,
        )
        scaled_data = neighbours_generator.generate_neighbours(
            num_samples=num_samples, sample_around_instance=self.sample_around_instance
        )
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
            norm=norm,
            verbose=verbose,
        )

    def solve_opt_problem(
        self,
        predict_fn: Callable,
        num_samples: int,
        weights: np.ndarray,
        H0: np.ndarray,
        scaled_data: np.ndarray,
        norm: Union[float, str],
        verbose: float,
    ) -> Tuple[np.ndarray, float]:
        """Solves the convex problem proposed in: https://arxiv.org/pdf/2003.08371.pdfF

        Args:
            predict_fn (Callable): function to compute the cumulative hazard.
            num_samples (int): number of neighbours.
            weights (np.ndarray): distance weights computed for each data point.
            H0 (np.ndarray): baseline cumulative hazard.
            scaled_data (np.ndarray): original data point and the computed neighbours.
            norm (Union[float, str]: number of the norm to be computed in the cvx problem.
            verbose (float): activate verbosity of the cvxpy solver.

        Returns:
            b.values (np.ndarray): obtained weights from the convex problem.
            result (float): residual value of the convex problem.
        """
        epsilon = 0.00000001
        num_features = scaled_data.shape[1]
        m = len(set(self.train_times))
        # To do: validate H_i_j_wc
        H_i_j_wc = predict_fn(scaled_data)
        times_to_fill = list(set(self.train_times))
        times_to_fill.sort()
        H_i_j_wc = np.array(
            [
                np.interp(times_to_fill, self.model_output_times, H_i_j_wc[i])
                for i in range(H_i_j_wc.shape[0])
            ]
        )
        log_correction = np.divide(H_i_j_wc, np.log(H_i_j_wc + epsilon))

        # Varible to look for
        b = cp.Variable((num_features, 1))

        # Reshape and log of predictions
        H = np.reshape(np.array(H_i_j_wc), newshape=(num_samples, m))
        LnH = np.log(H + epsilon)

        # Log of baseline cumulative hazard
        LnH0 = np.log(H0 + epsilon)
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

        opt_maker = OptFuncionMaker(E, w, logs, delta_t)
        funct = opt_maker.compute_function(norm=norm)

        objective = cp.Minimize(funct)
        prob = cp.Problem(objective)
        result = prob.solve(verbose=verbose)
        return b.value, result  # H_i_j_wc, weights, log_correction, scaled_data,
