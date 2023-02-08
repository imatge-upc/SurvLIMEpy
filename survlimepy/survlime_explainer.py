import logging
from copy import deepcopy
from typing import Callable, Tuple, Union, List, Literal, Optional
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from sksurv.functions import StepFunction
from survlimepy.utils.optimisation import OptFuncionMaker
from survlimepy.utils.neighbours_generator import NeighboursGenerator


class SurvLimeExplainer:
    """Look for the coefficient of a COX model."""

    def __init__(
        self,
        training_features: Union[np.ndarray, pd.DataFrame],
        training_events: Union[np.ndarray, pd.Series, List[Union[bool, float, int]]],
        training_times: Union[np.ndarray, pd.Series, List[Union[float, int]]],
        model_output_times: Optional[np.ndarray] = None,
        H0: Optional[Union[np.ndarray, List[float], StepFunction]] = None,
        kernel_width: Optional[float] = None,
        functional_norm: Union[float, str] = 2,
        random_state: Optional[int] = None,
    ) -> None:
        """Init function.

        Args:
            training_features (Union[np.ndarray, pd.DataFrame]): data used to train the bb model.
            training_events (Union[np.ndarray, pd.Series, List[Union[bool, float, int]]]): training events indicator.
            training_times (Union[np.ndarray, pd.Series, List[Union[float, int]]]): training times to event.
            model_output_times (Optional[np.ndarray]): output times of the bb model.
            H0 (Optional[Union[np.ndarray, List[float], StepFunction]]): baseline cumulative hazard.
            kernel_width (Optional[List[float]]): width of the kernel to be used to generate the neighbours and to compute distances.
            functional_norm (Optional[Union[float, str]]): functional norm to calculate the distance between the Cox model and the black box model.
            random_state (Optional[int]): number to be used for random seeds.

        Returns:
            None.
        """
        self.random_state = check_random_state(random_state)
        self.training_features = training_features
        self.training_events = training_events
        self.training_times = training_times
        self.model_output_times = model_output_times
        self.computed_weights = None
        self.montecarlo_weights = None
        self.is_data_frame = isinstance(self.training_features, pd.DataFrame)
        self.is_np_array = isinstance(self.training_features, np.ndarray)
        if not (self.is_data_frame or self.is_np_array):
            raise TypeError(
                "training_features must be either a numpy array or a pandas DataFrame."
            )
        if self.is_data_frame:
            self.feature_names = self.training_features.columns
            self.training_features_np = training_features.to_numpy()
        else:
            self.feature_names = [
                f"feature_{i}" for i in range(self.training_features.shape[1])
            ]
            self.training_features_np = np.copy(training_features)
        self.H0 = H0
        self.num_individuals = self.training_features.shape[0]
        self.num_features = self.training_features.shape[1]

        if kernel_width is None:
            num_sigma_opt = 4
            den_sigma_opt = self.num_individuals * (self.num_features + 2)
            pow_sigma_opt = 1 / (self.num_features + 4)
            kernel_default = (num_sigma_opt / den_sigma_opt) ** pow_sigma_opt
            self.kernel_width = kernel_default

        self.functional_norm = functional_norm

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
        max_difference_time_allowed: Optional[float] = None,
        max_hazard_value_allowed: Optional[float] = None,
        verbose: bool = False,
    ) -> np.ndarray:
        """Generates explanations for a prediction.

        Args:
            data_row (Union[List[float], np.ndarray, pd.Series]): data point to be explained.
            predict_fn (Callable): function that computes cumulative hazard.
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function.
            num_samples (int): number of neighbours to use.
            max_difference_time_allowed (Optional[float]): maximum difference between times allowed. If a difference exceeds this value, then max_difference_time_allowed will be used.
            max_hazard_value_allowed (Optional[float]): maximum hazard value allowed. If a prediction exceeds this value, then max_hazard_value_allows will be used.
            verbose (bool): whether or not to show cvxpy messages.

        Returns:
            cox_values (np.ndarray): obtained weights from the convex problem.
        """
        # To be used while plotting
        if isinstance(data_row, list):
            self.data_point = np.array(data_row).reshape(1, -1)
        elif isinstance(data_row, np.ndarray):
            total_dimensions_data_row = len(data_row.shape)
            total_rows = data_row.shape[0]
            if total_dimensions_data_row == 1:
                self.data_point = np.reshape(data_row, newshape=(1, -1))
            elif total_dimensions_data_row == 2:
                if total_rows > 1:
                    raise ValueError("data_point must contain a single row.")
                self.data_point = data_row
            else:
                raise ValueError("data_point must not have more than 2 dimensions.")
        elif isinstance(data_row, pd.Series):
            self.data_point = data_row.to_numpy().reshape(1, -1)
        else:
            raise ValueError(
                "data_point must be a list, a numpy array or a pandas Series."
            )

        # Generate neighbours
        neighbours_generator = NeighboursGenerator(
            training_features=self.training_features_np,
            data_row=self.data_point,
            sigma=self.kernel_width,
            random_state=self.random_state,
        )

        scaled_data = neighbours_generator.generate_neighbours(num_samples=num_samples)
        scaled_data_transformed = self.transform_data(data=scaled_data)

        # Solve optimisation problem
        opt_funcion_maker = OptFuncionMaker(
            training_features=self.training_features_np,
            training_events=self.training_events,
            training_times=self.training_times,
            kernel_width=self.kernel_width,
            neighbours=scaled_data,
            neighbours_transformed=scaled_data_transformed,
            num_samples=num_samples,
            data_point=self.data_point,
            predict_fn=predict_fn,
            type_fn=type_fn,
            functional_norm=self.functional_norm,
            model_output_times=self.model_output_times,
            H0=self.H0,
            max_difference_time_allowed=max_difference_time_allowed,
            max_hazard_value_allowed=max_hazard_value_allowed,
            verbose=verbose,
        )
        b = opt_funcion_maker.solve_problem()
        self.computed_weights = np.copy(b)
        return b

    def plot_weights(
        self,
        figsize: Tuple[int, int] = (10, 10),
        feature_names: Optional[List[str]] = None,
        scale_with_data_point: bool = False,
        figure_path: Optional[str] = None,
        with_colour: bool = True,
    ) -> None:
        # Create docstring of the function
        """Plots the weights of the computed COX model.

        Args:
            figsize (Tuple[int, int]): size of the figure.
            feature_names (Optional[List[str]]): names of the features.
            scale_with_data_point (bool): whether to perform the elementwise multiplication between the point to be explained and the coefficients.
            figure_path (Optional[str]): path to save the figure.
            with_colour (bool): boolean indicating whether the colour palette for positive coefficients is different than thecolour palette for negative coefficients. Default is set to True.

        Returns:
            None.
        """
        if self.computed_weights is None:
            raise ValueError(
                "SurvLIME weights not computed yet. Call explain_instance first before using this function."
            )
        else:
            are_there_any_nan = np.isnan(self.computed_weights).any()
            if are_there_any_nan:
                raise ValueError("Some of the coefficients contain nan values.")

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
        all_data = []
        for label, weights_separated, palette in zip(
            [pos_feature_names, neg_feature_names],
            [pos_weights, neg_weights],
            ["Reds_r", "Blues"],
        ):
            data = pd.DataFrame({"features": label, "weights": weights_separated})
            all_data.append(data)
            if with_colour:
                ax.bar(
                    "features",
                    "weights",
                    data=data,
                    color=sns.color_palette(palette, n_colors=len(label)),
                    label=label,
                )
        if not with_colour:
            data = pd.concat(all_data)
            ax.bar(
                "features",
                "weights",
                data=data,
                color="grey",
                label=[*pos_feature_names, *neg_feature_names],
            )
        ax.set_xlabel("Features", fontsize=14)
        ax.set_ylabel("SurvLIME value", fontsize=14)
        ax.set_title("Feature importance", fontsize=16, fontweight="bold")

        ax.tick_params(axis="x", labelsize=14, rotation=90)
        ax.tick_params(axis="y", labelsize=14)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if figure_path is not None:
            plt.savefig(figure_path, dpi=200, bbox_inches="tight")

        plt.show()

    def montecarlo_explanation(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray, List],
        predict_fn: Callable,
        type_fn: Literal["survival", "cumulative"] = "cumulative",
        num_samples: int = 1000,
        num_repetitions: int = 10,
        max_difference_time_allowed: Optional[float] = None,
        max_hazard_value_allowed: Optional[float] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Generates explanations for a prediction.

        Args:
            data (Union[pd.DataFrame, pd.Series, np.ndarray, List]): data points to be explained.
            predict_fn (Callable): function that computes cumulative hazard.
            type_fn (Literal["survival", "cumulative"]): whether predict_fn is the cumulative hazard funtion or survival function.
            num_samples (int): number of neighbours to use.
            num_repetitions (int): number of times to repeat the explanation.
            max_difference_time_allowed (Optional[float]): maximum difference between times allowed. If a difference exceeds this value, then max_difference_time_allowed will be used.
            max_hazard_value_allowed (Optional[float]): maximum hazard value allowed. If a prediction exceeds this value, then max_hazard_value_allows will be used.
            verbose (bool): whether or not to show cvxpy messages.

        Returns:
            montecarlo_explanation (pd.DataFrame): dataframe with the montecarlo explanation.
        """
        if isinstance(data, pd.DataFrame):
            data_transformed = data.to_numpy()
        elif isinstance(data, pd.Series):
            data_transformed = data.to_numpy().reshape(1, -1)
        elif isinstance(data, np.ndarray):
            n_dim = len(data.shape)
            if n_dim == 1:
                n_rows = 1
            else:
                n_rows = data.shape[0]
            data_transformed = np.reshape(data, newshape=(n_rows, -1))
        elif isinstance(data, list):
            data_transformed = np.array(data).reshape(1, -1)
        else:
            raise ValueError(
                "data must be a pandas DataFrame, a pandas Series, a numpy array or a list."
            )
        self.matrix_to_explain = deepcopy(data_transformed)
        all_solved = True
        total_rows = data_transformed.shape[0]
        total_cols = data_transformed.shape[1]
        weights = np.empty(shape=(total_rows, total_cols))
        for i_row in tqdm(range(total_rows)):
            current_row = data_transformed[i_row]
            weights_current_row = []
            i_rep = 0
            while i_rep < num_repetitions:
                # sample data point from the dataset
                b = self.explain_instance(
                    data_row=current_row,
                    predict_fn=predict_fn,
                    type_fn=type_fn,
                    num_samples=num_samples,
                    max_difference_time_allowed=max_difference_time_allowed,
                    max_hazard_value_allowed=max_hazard_value_allowed,
                    verbose=verbose,
                )
                are_there_any_nan_value = np.isnan(b).any()

                if are_there_any_nan_value:
                    all_solved = False
                else:
                    weights_current_row.append(b)
                    i_rep += 1

            weights_current_row = np.array(weights_current_row)
            mean_weight_row = np.mean(weights_current_row, axis=0)
            weights[i_row] = mean_weight_row

        if not all_solved:
            logging.warning(
                "There were some simulations without a solution. Try to run it with verbose=True."
            )
        self.montecarlo_weights = weights
        return weights

    def plot_montecarlo_weights(
        self,
        figsize: Tuple[int, int] = (10, 10),
        feature_names: Optional[List[str]] = None,
        scale_with_data_point: bool = False,
        figure_path: Optional[str] = None,
        with_colour: bool = True,
    ) -> None:
        """Generates explanations for a prediction.

        Args:
            figsize (Tuple[int, int]): size of the figure.
            feature_names Optional[List[str]]): names of the features.
            scale_with_data_point (bool): whether to perform the elementwise multiplication between the point to be explained and the coefficients.
            figure_path (Optional[str]): path to save the figure.
            with_colour (bool): boolean indicating whether the colour palette for positive coefficients is different than thecolour palette for negative coefficients. Default is set to True.

        Returns:
            None.
        """
        if self.montecarlo_weights is None:
            raise ValueError(
                "Monte-Carlo weights not computed yet. Call montecarlo_explanation first before using this function."
            )
        else:
            are_there_any_nan = np.isnan(self.montecarlo_weights).any()
            if are_there_any_nan:
                raise ValueError("Some of the coefficients contain nan values.")

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
        if with_colour:
            p = sns.boxenplot(
                x="variable",
                y="value",
                data=data_melt,
                palette=custom_pal,
                ax=ax,
            )
        else:
            p = sns.boxenplot(
                x="variable",
                y="value",
                data=data_melt,
                color="grey",
                ax=ax,
            )
        ax.tick_params(labelrotation=90)
        p.set_xlabel("Features", fontsize=14)
        p.set_ylabel("SurvLIME value", fontsize=14)
        p.yaxis.grid(True)
        p.xaxis.grid(True)

        p.set_title("Feature importance", fontsize=16, fontweight="bold")

        plt.xticks(fontsize=16, rotation=90)
        plt.yticks(fontsize=14, rotation=0)

        if figure_path is not None:
            plt.savefig(figure_path, dpi=200, bbox_inches="tight")

        plt.show()
