from typing import Union, Callable, Optional, Literal, List
from sklearn.metrics import pairwise_distances
from survlimepy.utils.predict import predict_wrapper
from sksurv.nonparametric import nelson_aalen_estimator
from sksurv.functions import StepFunction
import numpy as np
import cvxpy as cp
import pandas as pd
import logging
from survlimepy.utils.step_function import transform_step_function


class OptFuncionMaker:
    """This class manages the optimisation process."""

    def __init__(
        self,
        training_features: np.ndarray,
        training_events: Union[np.ndarray, pd.Series, List[Union[bool, float, int]]],
        training_times: Union[np.ndarray, pd.Series, List[Union[float, int]]],
        kernel_width: float,
        neighbours: np.ndarray,
        neighbours_transformed: Union[np.ndarray, pd.DataFrame],
        num_samples: int,
        data_point: np.ndarray,
        predict_fn: Callable,
        type_fn: Literal["survival", "cumulative"],
        functional_norm: Union[float, str],
        model_output_times: Optional[np.ndarray] = None,
        H0: Optional[Union[np.ndarray, List[float], StepFunction]] = None,
        max_difference_time_allowed: Optional[float] = None,
        max_hazard_value_allowed: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        """Init function.

        Args:
            training_features (np.ndarray): data used to train the bb model.
            training_events (Union[np.ndarray, pd.Series, List[Union[bool, float, int]]]): training events indicator.
            training_times (Union[np.ndarray, pd.Series, List[Union[float, int]]]): training times to event.
            kernel_width (float): width of the kernel to be used to compute distances.
            neighbours (np.ndarray): neighbours (num_samples x features).
            neighbours_transformed (Union[np.ndarray, pd.DataFrame]): neighbours in the appropriate format to use the prediction function.
            num_samples (int): number of neighbours to use.
            data_point (np.ndarray): data point to be explained.
            predict_fn (Callable): function that computes cumulative hazard.
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function.
            functional_norm (Union[float, str]): functional norm to calculate the distance between the Cox model and the black box model.
            model_output_times (Optional[np.ndarray]): output times of the bb model.
            H0 (Optional[Union[np.ndarray, List[float], StepFunction]]): baseline cumulative hazard.
            max_difference_time_allowed (Optional[float]): maximum difference between times allowed. If a difference exceeds this value, then max_difference_time_allowed will be used.
            max_hazard_value_allowed (Optional[float]): maximum hazard value allowed. If a prediction exceeds this value, then max_hazard_value_allows will be used.
            verbose (bool): whether or not to show cvxpy messages.

        Returns:
            None.
        """
        self.training_features = training_features
        self.validate_events_times(training_events)
        self.validate_events_times(training_times)
        if isinstance(training_events, pd.Series):
            self.training_events = training_events.to_numpy()
        else:
            self.training_events = training_events
        if isinstance(training_times, pd.Series):
            self.training_times = training_times.to_numpy()
        else:
            self.training_times = training_times
        self.kernel_width = kernel_width
        self.neighbours = neighbours
        self.neighbours_transformed = neighbours_transformed
        self.num_samples = num_samples
        self.data_point = data_point
        self.predict_fn = predict_fn

        if type_fn not in ["survival", "cumulative"]:
            raise ValueError("type_fn must be either survival or cumulative string.")
        self.type_fn = type_fn

        self.unique_times_to_event = np.sort(np.unique(self.training_times))
        self.m = self.unique_times_to_event.shape[0]

        if model_output_times is None:
            self.model_output_times = np.copy(self.unique_times_to_event)
        else:
            self.model_output_times = model_output_times

        if (
            isinstance(functional_norm, float) or isinstance(functional_norm, int)
        ) and functional_norm < 1:
            raise ValueError(
                f"functional_norm should be greater than 1, given value {functional_norm}."
            )
        elif isinstance(functional_norm, str) and functional_norm != "inf":
            raise ValueError(f"Invalid string for functional_norm. It must be \"inf\".")
        if not (
            isinstance(functional_norm, float)
            or isinstance(functional_norm, int)
            or isinstance(functional_norm, str)
        ):
            raise TypeError(f"functional_norm must be int, float or str type.")
        self.functional_norm = functional_norm

        if H0 is None:
            self.H0 = self.compute_nelson_aalen_estimator(
                self.training_events, self.training_times
            )
        else:
            if isinstance(H0, list):
                self.H0 = np.array(H0).reshape(-1, 1)
            elif isinstance(H0, np.ndarray):
                total_dimensions_H0 = len(H0.shape)
                if total_dimensions_H0 == 1:
                    self.H0 = np.reshape(H0, newshape=(-1, 1))
                elif total_dimensions_H0 == 2:
                    if self.H0.shape[1] > 1:
                        raise ValueError("H0 must contain a single column.")
                    self.H0 = H0
                else:
                    raise ValueError("H0 must be an array of maximum 2 dimensions.")
            elif isinstance(H0, StepFunction):
                self.H0 = transform_step_function(
                    array_step_functions=np.array([H0]), return_column_vector=True
                )
            else:
                raise ValueError(
                    "H0 must be a list, a numpy array or a sksurv StepFunction."
                )

        if max_difference_time_allowed is not None:
            if max_difference_time_allowed < 0:
                raise ValueError("max_difference_time_allowed must be positive.")
        self.max_difference_time_allowed = max_difference_time_allowed

        if max_hazard_value_allowed is not None:
            if max_hazard_value_allowed < 0:
                raise ValueError("max_hazard_value_allowed must be positive.")
        self.max_hazard_value_allowed = max_hazard_value_allowed

        self.verbose = verbose

        if self.H0.shape[0] != self.m:
            raise ValueError(f"H0 must have {self.m} rows/elements.")
        if self.H0.shape[1] != 1:
            raise ValueError("H0 must have 1 column.")

        self.limit_H_warning = 500
        self.epsilon = 10 ** (-6)
        self.sd_features = np.std(
            self.training_features, axis=0, dtype=self.training_features.dtype
        )
        self.sd_sq = np.square(self.sd_features)

    @staticmethod
    def validate_events_times(vector):
        if not (
            isinstance(vector, list)
            or isinstance(vector, np.ndarray)
            or isinstance(vector, pd.Series)
        ):
            raise TypeError(
                "Both training_events and training_times must be a list, a numpy array or a pandas Series."
            )
        if isinstance(vector, np.ndarray) and len(vector) > 1:
            raise TypeError(
                "Both training_events and training_times must be 1D numpy arrays"
            )
        return None

    @staticmethod
    def compute_nelson_aalen_estimator(
        event: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        """Compute Nelson-Aalen estimator.

        Args:
            event (np.ndarray): array of events.
            time (np.ndarray): array of times.

        Returns:
            H0 (np.ndarray): the Nelson-Aalen estimator
        """
        nelson_aalen = nelson_aalen_estimator(event, time)
        H0 = nelson_aalen[1]
        m = H0.shape[0]
        H0 = np.reshape(H0, newshape=(m, 1))
        return H0

    def weighted_euclidean_distance(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the weighted Euclidean distance.

        Args:
            x (np.ndarray): the first array.
            y (np.ndarray): the second array.

        Returns:
            distance (np.ndarray): the distance between the two arrays.
        """
        diff = x - y
        diff_sq = np.square(diff)
        div = np.divide(diff_sq, self.sd_sq)
        distance = np.sqrt(np.sum(div))
        return distance

    def kernel_fn(self, d):
        """Compute the kernel function.

        Args:
            d (np.ndarray): the distance.

        Returns:
            weight (np.ndarray): the kernel weight.
        """
        num = -(d**2)
        den = 2 * self.kernel_width
        weight = np.exp(num / den)
        return weight

    def get_weights(self):
        """Compute the weights of each individual.

        Args:
            None

        Returns:
            w (np.ndarray): the weights for each individual.
        """
        # Compute weights.
        distances = pairwise_distances(
            self.neighbours, self.data_point, metric=self.weighted_euclidean_distance
        ).ravel()
        weights = self.kernel_fn(distances)
        w = np.reshape(weights, newshape=(self.num_samples, 1))
        return w

    def get_predictions(self) -> np.ndarray:
        """Compute the prediction for each neighbour.

        Args:
            None

        Returns:
            H (np.ndarray): the prediction.
        """
        # Compute predictions
        FN_pred = predict_wrapper(
            predict_fn=self.predict_fn,
            data=self.neighbours_transformed,
            unique_times_to_event=self.unique_times_to_event,
            model_output_times=self.model_output_times,
        )
        if self.type_fn == "survival":
            H_score = -np.log(FN_pred + self.epsilon)
        else:
            H_score = np.copy(FN_pred)
        max_H_score = np.max(H_score)
        if self.max_hazard_value_allowed is None and max_H_score > self.limit_H_warning:
            logging.warning(
                f"The prediction function produces extreme values: {max_H_score}. In terms of survival, Pr(Survival) is almost 0. Try to set max_hazard_value_allowed parameter to clip this value."
            )
        if self.max_hazard_value_allowed is not None:
            H_score = np.clip(
                a=H_score, a_min=None, a_max=self.max_hazard_value_allowed
            )
        H = np.reshape(np.array(H_score), newshape=(self.num_samples, self.m))
        return H

    def get_delta_t(self) -> np.ndarray:
        """Compute the vector of delta times.

        Args:
            None

        Returns:
            delta_t (np.ndarray): the vector of delta times.
        """
        t = np.empty(shape=(self.m + 1, 1))
        t[: self.m, 0] = self.unique_times_to_event
        t[self.m, 0] = t[self.m - 1, 0] + self.epsilon
        delta_t = [t[i + 1, 0] - t[i, 0] for i in range(self.m)]
        delta_t = np.reshape(np.array(delta_t), newshape=(self.m, 1))
        if self.max_difference_time_allowed is not None:
            delta_t = np.clip(
                a=delta_t, a_min=None, a_max=self.max_difference_time_allowed
            )
        return delta_t

    def compute_norm(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the norm of a given matrix.

        Args:
            matrix (np.ndarray): the matrix over which to compute the norm.

        Returns:
            matrix_norm (np.ndarray): The norm of the matrix
        """
        if isinstance(self.functional_norm, str):
            matrix_norm = cp.norm(matrix, "inf")
        elif self.functional_norm == 1:
            matrix_norm = cp.abs(matrix)
        elif self.functional_norm % 2 != 0:
            matrix_abs = cp.abs(matrix)
            matrix_norm = cp.power(matrix_abs, self.functional_norm)
        else:
            matrix_norm = cp.power(matrix, self.functional_norm)
        return matrix_norm

    def solve_problem(self) -> np.ndarray:
        """Solve the optimisation problem.

        Args:
            None.

        Returns:
            cox_coefficients (np.ndarray): The result of the optimisation problem.
        """
        # Varible to look for
        num_features = self.neighbours.shape[1]
        cox_coefficients = np.full(shape=num_features, fill_value=np.nan)
        b = cp.Variable((num_features, 1))

        # Get predictions
        H = self.get_predictions()
        LnH = np.log(H + self.epsilon)

        # Compute the log correction
        log_correction = np.divide(H, LnH)

        # Log of baseline cumulative hazard
        LnH0 = np.log(self.H0 + self.epsilon)

        # Weights of individuals
        w = self.get_weights()

        # Time differences
        delta_t = self.get_delta_t()

        # Matrices to produce the proper sizes
        ones_N = np.ones(shape=(self.num_samples, 1))
        ones_m_1 = np.ones(shape=(self.m, 1))
        B = np.dot(ones_N, LnH0.T)
        C = LnH - B
        Z = self.neighbours @ b
        D = Z @ ones_m_1.T
        E = C - D
        E_norm = self.compute_norm(matrix=E)
        V_sq = cp.square(log_correction)
        F = cp.multiply(E_norm, V_sq)
        G = F @ delta_t
        funct = G.T @ w

        # Solving the problem
        try:
            objective = cp.Minimize(funct)
            prob = cp.Problem(objective)
            prob.solve(solver="OSQP", verbose=self.verbose, eps_abs=1e-3, eps_rel=1e-3)
            if prob.status not in ["infeasible", "unbounded"]:
                cox_coefficients = b.value[:, 0]
            else:
                logging.warning(f"The status of the problem is {prob.status}.")
        except (
            cp.error.DCPError,
            cp.error.DGPError,
            cp.error.DPPError,
            cp.error.SolverError,
        ) as e:
            msg = f"{str(e)}\n Returning an array full of nan values."
            logging.warning(msg)
        return cox_coefficients
