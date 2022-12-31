from functools import partial
import logging
from copy import deepcopy
from typing import Callable, Tuple, Union, List, Literal
import numpy as np
import pandas as pd
import cvxpy as cp
import sklearn
import sklearn.preprocessing
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from sksurv.nonparametric import nelson_aalen_estimator
from sksurv.functions import StepFunction
from survlimepy.utils.optimization import OptFuncionMaker
from survlimepy.utils.neighbours_generator import NeighboursGenerator
from survlimepy.utils.predict import predict_wrapper
from survlimepy.utils.step_function import transform_step_function


class SurvLimeExplainer:
    """
    Look for the coefficient of a COX model."""

    def __init__(
        self,
        training_features: Union[np.ndarray, pd.DataFrame],
        training_events: List[Union[float, int]],
        training_times: List[Union[bool, float, int]],
        model_output_times: np.ndarray,
        categorical_features: List[int] = None,
        H0: Union[np.ndarray, List[float], StepFunction] = None,
        kernel_width: float = None,
        kernel_distance: str = "euclidean",
        kernel_fn: Callable = None,
        functional_norm: Union[float, str] = 2,
        sample_around_instance: bool = False,
        random_state: int = None,
    ) -> None:
        """Init function.
        Args:
            training_features (Union[np.ndarray, pd.DataFrame]): data used to train the bb model.
            training_events (List[Union[float, int]]): training events indicator.
            training_times (List[Union[bool, float, int]]): training times to event.
            model_output_times (np.ndarray): output times of the bb model.
            categorical_features (List[int]): list of integers indicating the categorical features.
            H0 (Union[np.ndarray, List[float]], StepFunction): baseline cumulative hazard.
            kernel_width (float): width of the kernel to be used for computing distances.
            kernel_distance (str): metric to be used for computing neighbours distance to the original point.
            kernel_fn (Callable): kernel function to be used for computing distances.
            functional_norm (Union[float, str]): functional norm to calculate the distance between the Cox model and the black box model.
            sample_around_instance (bool): whether we sample around instances or not.
            random_state (int): number to be used for random seeds.
        Returns:
            None.
        """

        self.random_state = check_random_state(random_state)
        self.sample_around_instance = sample_around_instance
        self.training_features = training_features
        self.training_events = training_events
        self.training_times = training_times
        self.unique_times_to_event = np.sort(np.unique(self.training_times))
        self.categorical_features = categorical_features
        self.model_output_times = model_output_times
        self.computed_weights = None
        self.montecarlo_weights = None
        self.is_data_frame = isinstance(self.training_features, pd.DataFrame)
        if self.is_data_frame:
            self.feature_names = self.training_features.columns
        else:
            self.feature_names = [
                f"feature_{i}" for i in range(self.training_features.shape[1])
            ]

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
                    self.H0 = H0
                else:
                    raise ValueError("H0 must be an array of maximum 2 dimensions.")
            elif isinstance(H0, StepFunction):
                self.H0 = transform_step_function(
                    array_step_functions=np.array([H0]), return_column_vector=True
                )
            else:
                raise ValueError("H0 must be a list, a numpy array or a StepFunction.")

        m = self.unique_times_to_event.shape[0]
        if self.H0.shape[0] != m:
            raise ValueError(f"H0 must have {m} rows/elements.")
        if self.H0.shape[1] != 1:
            raise ValueError("H0 must have 1 column.")
        if kernel_width is None:
            kernel_width = np.sqrt(self.training_features.shape[1]) * 0.75
        kernel_width = float(kernel_width)

        self.kernel_distance = kernel_distance

        if kernel_fn is None:

            def kernel_fn(d: np.ndarray, kernel_width: float) -> np.ndarray:
                return np.sqrt(np.exp(-(d**2) / kernel_width**2))

        self.kernel_fn = partial(kernel_fn, kernel_width=kernel_width)
        self.functional_norm = functional_norm
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(self.training_features)

    @staticmethod
    def compute_nelson_aalen_estimator(
        event: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        nelson_aalen = nelson_aalen_estimator(event, time)
        H0 = nelson_aalen[1]
        m = H0.shape[0]
        H0 = np.reshape(H0, newshape=(m, 1))
        return H0

    def transform_data(
        self, data: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Transforms data according with what the model needs.
        Args:
            data (Union[np.ndarray, pd.DataFrame]): data to be transformed.

        Returns:
            data_transformed (Union[np.ndarray, pd.DataFrame]): transformed data.
        """
        if self.is_data_frame:
            data_transformed = pd.DataFrame(data, columns=self.feature_names)
        else:
            training_dtype = self.training_features.dtype
            data_transformed = np.copy(data)
            data_transformed = data_transformed.astype(training_dtype)
        return data_transformed

    def explain_instance(
        self,
        data_row: Union[List[float], np.ndarray, pd.Series],
        predict_fn: Callable,
        type_fn: Literal["survival", "cumulative"] = "cumulative",
        num_samples: int = 1000,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Generates explanations for a prediction.
        Args:
            data_row (Union[List[float], np.ndarray, pd.Series]): data point to be explained.
            predict_fn (Callable): function that computes cumulative hazard.
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function.
            num_samples (int): number of neighbours to use.
            verbose (bool): whether or not to show cvxpy messages.
        Returns:
            cox_values (np.ndarray): obtained weights from the convex problem.
        """
        # To be used while plotting
        if isinstance(data_row, list):
            self.data_point = np.array(data_row).reshape(1, -1)
        elif isinstance(data_row, np.ndarray):
            total_dimensions_data_row = len(data_row.shape)
            if total_dimensions_data_row == 1:
                self.data_point = np.reshape(data_row, newshape=(1, -1))
            elif total_dimensions_data_row == 2:
                self.data_point = data_row
            else:
                raise ValueError("data_point must not have more than 2 dimensions.")
        elif isinstance(data_row, pd.Series):
            self.data_point = data_row.to_numpy().reshape(1, -1)
        else:
            raise ValueError("data_point must be either a list or a numpy array.")

        neighbours_generator = NeighboursGenerator(
            training_features=self.training_features,
            data_row=self.data_point,
            categorical_features=self.categorical_features,
            random_state=self.random_state,
        )

        scaled_data = neighbours_generator.generate_neighbours(
            num_samples=num_samples, sample_around_instance=self.sample_around_instance
        )

        scaled_data_transformed = self.transform_data(data=scaled_data)

        distances = sklearn.metrics.pairwise_distances(
            scaled_data, self.data_point, metric=self.kernel_distance
        ).ravel()

        weights = self.kernel_fn(distances)

        # Solution for the optimisation problems
        cox_values = self.solve_opt_problem(
            predict_fn=predict_fn,
            type_fn=type_fn,
            num_samples=num_samples,
            weights=weights,
            H0=self.H0,
            data=scaled_data_transformed,
            norm=self.functional_norm,
            verbose=verbose,
        )
        return cox_values

    def solve_opt_problem(
        self,
        predict_fn: Callable,
        type_fn: Literal["survival", "cumulative"],
        num_samples: int,
        weights: np.ndarray,
        H0: np.ndarray,
        data: np.ndarray,
        norm: Union[float, str],
        verbose: bool,
    ) -> Tuple[np.ndarray, float]:
        """Solves the convex problem proposed in: https://arxiv.org/pdf/2003.08371.pdf
        Args:
            predict_fn (Callable): function to compute the cumulative hazard.
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function.
            num_samples (int): number of neighbours.
            weights (np.ndarray): distance weights computed for each data point.
            H0 (np.ndarray): baseline cumulative hazard.
            data (np.ndarray): neighbours generated.
            norm (Union[float, str]: functional norm to calculate the distance between the Cox model and the black box model.
            verbose (bool): activate verbosity of the cvxpy solver.

        Returns:
            cox_coefficients (np.ndarray): coefficients of a COX model.
        """
        epsilon = 10 ** (-6)
        num_features = data.shape[1]
        m = self.unique_times_to_event.shape[0]
        FN_pred = predict_wrapper(
            predict_fn=predict_fn,
            data=data,
            unique_times_to_event=self.unique_times_to_event,
            model_output_times=self.model_output_times,
        )

        # From now on, original data must be a numpy array
        if isinstance(data, np.ndarray):
            data_np = data
        elif isinstance(data, pd.DataFrame):
            data_np = data.to_numpy()
        else:
            raise TypeError(
                f"Unknown data type {type(data)} only np.ndarray or pd.DataFrame allowed."
            )

        # Varible to look for
        b = cp.Variable((num_features, 1))

        # Reshape and log of predictions
        if type_fn == "survival":
            H_score = -np.log(FN_pred + epsilon)
        elif type_fn == "cumulative":
            H_score = deepcopy(FN_pred)
        else:
            raise ValueError("type_fn must be either survival or cumulative string.")
        log_correction = np.divide(H_score, np.log(H_score + epsilon))
        H = np.reshape(np.array(H_score), newshape=(num_samples, m))
        LnH = np.log(H + epsilon)

        # Log of baseline cumulative hazard
        LnH0 = np.log(H0 + epsilon)
        # Compute the log correction
        logs = np.reshape(log_correction, newshape=(num_samples, m))

        # Distance weights
        w = np.reshape(weights, newshape=(num_samples, 1))

        # Time differences
        t = np.empty(shape=(m + 1, 1))
        t[:m, 0] = self.unique_times_to_event
        t[m, 0] = t[m - 1, 0] + epsilon
        delta_t = [t[i + 1, 0] - t[i, 0] for i in range(m)]
        delta_t = np.reshape(np.array(delta_t), newshape=(m, 1))

        # Matrices to produce the proper sizes
        ones_N = np.ones(shape=(num_samples, 1))
        ones_m_1 = np.ones(shape=(m, 1))
        B = np.dot(ones_N, LnH0.T)
        C = LnH - B
        Z = data_np @ b
        D = Z @ ones_m_1.T
        E = C - D

        opt_maker = OptFuncionMaker(E, w, logs, delta_t)
        funct = opt_maker.compute_function(norm=norm)

        objective = cp.Minimize(funct)
        prob = cp.Problem(objective)
        result = prob.solve(verbose=verbose)
        cox_coefficients = b.value[:, 0]

        self.computed_weights = cox_coefficients
        return cox_coefficients

    def plot_weights(
        self,
        figsize: Tuple[int, int] = (10, 10),
        feature_names: List[str] = None,
        scale_with_data_point: bool = False,
        figure_path: str = None,
    ) -> None:
        # Create docstring of the function
        """Plots the weights of the computed COX model.
        Args:
            figsize (Tuple[int, int]): size of the figure.
            feature_names (List[str]): names of the features.
            scale_with_data_point (bool): whether to perform the elementwise multiplication between the point to be explained and the coefficients.
            figure_path (str): path to save the figure.
        Returns:
            None.
        """
        if self.computed_weights is None:
            raise ValueError(
                "SurvLIME weights not computed yet. Call explain_instance first before using this function."
            )

        if feature_names is not None:
            if len(feature_names) != self.computed_weights.shape[0]:
                raise TypeError(
                    f"feature_names must have {self.computed_weights[0]} elements."
                )
        else:
            feature_names = self.feature_names

        if scale_with_data_point:
            weights = self.computed_weights * self.data_point
        else:
            weights = self.computed_weights

        _, ax = plt.subplots(figsize=figsize)

        # sort weights in descending order
        idx_sort = np.argsort(weights)[::-1]
        sorted_weights = weights[idx_sort]
        # sort feature names so that they match the sorted weights
        sorted_feature_names = [feature_names[i] for i in idx_sort]

        # divide the sorted weights and sorted feature names into positive and negative
        pos_weights = [w for w in sorted_weights if w > 0]
        pos_feature_names = [
            f for f, w in zip(sorted_feature_names, sorted_weights) if w > 0
        ]
        neg_weights = [w for w in sorted_weights if w < 0]
        neg_feature_names = [
            f for f, w in zip(sorted_feature_names, sorted_weights) if w < 0
        ]

        for label, weights_separated, palette in zip(
            [pos_feature_names, neg_feature_names],
            [pos_weights, neg_weights],
            ["Reds_r", "Blues"],
        ):
            data = pd.DataFrame({"features": label, "weights": weights_separated})
            ax.bar(
                "features",
                "weights",
                data=data,
                color=sns.color_palette(palette, n_colors=len(label)),
                label=label,
            )

        ax.set_xlabel("Feature", fontsize=16)
        ax.set_ylabel("Weight", fontsize=16)
        ax.set_title("SurvLIME weights", fontsize=16)

        ax.tick_params(axis="x", labelsize=14, rotation=90)
        ax.tick_params(axis="y", labelsize=14)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if figure_path is not None:
            plt.savefig(figure_path, dpi=200, bbox_inches="tight")

        plt.show()

    def montecarlo_explanation(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        predict_fn: Callable,
        type_fn: Literal["survival", "cumulative"] = "cumulative",
        num_samples: int = 1000,
        num_repetitions: int = 10,
    ) -> pd.DataFrame:
        """Generates explanations for a prediction.
        Args:
            data (np.ndarray): data points to be explained.
            predict_fn (Callable): function that computes cumulative hazard.
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function.
            num_samples (int): number of neighbours to use.
            num_repetitions (int): number of times to repeat the explanation.
        Returns:
            montecarlo_explanation (pd.DataFrame): dataframe with the montecarlo explanation.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        self.matrix_to_explain = np.copy(data)
        all_solved = True
        total_rows = data.shape[0]
        total_cols = data.shape[1]
        weights = np.empty(shape=(total_rows, total_cols))
        for i_row in tqdm(range(total_rows)):
            current_row = data[i_row]
            weights_current_row = []
            for _ in range(num_repetitions):
                # sample data point from the dataset
                try:
                    b = self.explain_instance(
                        data_row=current_row,
                        predict_fn=predict_fn,
                        type_fn=type_fn,
                        num_samples=num_samples,
                        verbose=False,
                    )
                    weights_current_row.append(b)
                except (
                    cp.error.DCPError,
                    cp.error.DGPError,
                    cp.error.DPPError,
                    cp.error.SolverError,
                ):
                    all_solved = False
            weights_current_row = np.array(weights_current_row)
            mean_weight_row = np.mean(weights_current_row, axis=0)
            weights[i_row] = mean_weight_row

        if not all_solved:
            logging.warning(f"There were some simulations without a solution.")
        self.montecarlo_weights = weights
        return weights

    def plot_montecarlo_weights(
        self,
        figsize: Tuple[int, int] = (10, 10),
        feature_names: List[str] = None,
        scale_with_data_point: bool = False,
        figure_path: str = None,
    ) -> None:
        """Generates explanations for a prediction.
        Args:
            figsize (Tuple[int, int]): size of the figure.
            feature_names (List[str]): names of the features.
            scale_with_data_point (bool): whether to perform the elementwise multiplication between the point to be explained and the coefficients.
            figure_path (str): path to save the figure.
        Returns:
            None.
        """
        if self.montecarlo_weights is None:
            raise ValueError(
                "Monte-Carlo weights not computed yet. Call montecarlo_explanation first before using this function."
            )

        total_cols = self.montecarlo_weights.shape[1]

        if feature_names is not None:
            if len(feature_names) != total_cols:
                raise TypeError(f"feature_names must have {total_cols} elements.")
            col_names = feature_names
        else:
            col_names = self.feature_names

        if scale_with_data_point:
            scaled_data = np.multiply(self.montecarlo_weights, self.matrix_to_explain)

        else:
            scaled_data = self.montecarlo_weights
        data = pd.DataFrame(data=scaled_data, columns=col_names)

        sns.set()
        median_up = {}
        median_down = {}
        threshold = 0
        for (columnName, columnData) in data.items():
            median_value = np.median(columnData)
            if median_value > threshold:
                median_up[columnName] = median_value
            else:
                median_down[columnName] = median_value

        median_up = dict(
            sorted(median_up.items(), key=lambda item: item[1], reverse=True)
        )
        median_down = dict(
            sorted(median_down.items(), key=lambda item: item[1], reverse=True)
        )
        pal_up = sns.color_palette("Reds_r", n_colors=len(median_up))
        pal_down = sns.color_palette("Blues", n_colors=len(median_down))
        colors_up = {key: val for key, val in zip(median_up.keys(), pal_up)}
        colors_down = {key: val for key, val in zip(median_down.keys(), pal_down)}
        custom_pal = {**colors_up, **colors_down}
        data_reindex = data.reindex(columns=custom_pal.keys())
        data_melt = pd.melt(data_reindex)

        _, ax = plt.subplots(figsize=figsize)
        ax.tick_params(labelrotation=90)
        p = sns.boxenplot(
            x="variable",
            y="value",
            data=data_melt,
            palette=custom_pal,
            ax=ax,
        )
        ax.tick_params(labelrotation=90)
        p.set_xlabel("Features", fontsize=14, fontweight="bold")
        p.set_ylabel("SurvLIME value", fontsize=14, fontweight="bold")
        p.yaxis.grid(True)
        p.xaxis.grid(True)

        p.set_title("SurvLIME values", fontsize=16, fontweight="bold")

        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=14, rotation=0)

        if figure_path is not None:
            plt.savefig(figure_path, dpi=200, bbox_inches="tight")

        plt.show()
