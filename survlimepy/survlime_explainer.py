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

### Aditions needed forn the reviewer
from survlimepy.utils.predict import predict_wrapper
from sksurv.nonparametric import nelson_aalen_estimator
from sklearn.metrics import pairwise_distances
import cvxpy as cp
import timeit
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
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
        scalar_form: bool = False
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
        self.max_difference_time_allowed = max_difference_time_allowed
        self.max_hazard_value_allowed = max_hazard_value_allowed
        self.num_samples = num_samples
        self.type_fn = type_fn
        self.predict_fn = predict_fn
        self.limit_H_warning = 99
        
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


        if scalar_form:
            b = self.prueba(data_row,
                                    scaled_data_transformed,
                                    predict_fn,
                                    type_fn,
                                    num_samples,
                                    max_difference_time_allowed,
                                    max_hazard_value_allowed
                                    )
            
        else:
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

    def prueba(self, data,
                               neighbours,
                               predict_fn,
                               type_fn,
                               num_samples,
                               max_difference_time_allowed,
                               max_hazard_value_allowed) -> np.ndarray:
        
        epsilon = 0.00000001
        self.epsilon = epsilon
        self.neighbours = neighbours
        # Compute distances for the neighbours
        distances = pairwise_distances(
            neighbours, data, metric=self.weighted_euclidean_distance
        ).ravel()
        weights = self.kernel_fn(distances)
        weights = np.reshape(weights, newshape=(num_samples, 1))

        self.unique_times_to_event = np.sort(np.unique(self.training_times))
        self.m = self.unique_times_to_event.shape[0]

        if self.model_output_times is None:
            self.model_output_times = np.copy(self.unique_times_to_event)
        else:
            self.model_output_times = self.model_output_times
        
        self.neighbours_transformed = neighbours
        predictions = self.get_predictions()
        log_correction = [np.divide(np.array(x), np.log(np.array(x)+0.0001)) for x in predictions]

        nelson_aalen = nelson_aalen_estimator(self.training_events, self.training_times)
        H0 = nelson_aalen[1]
        m = H0.shape[0]
        H0 = np.reshape(H0, newshape=(m, 1))
        start_time = timeit.default_timer()

        
        num_features = data.shape[1]# Is there a nicer way to obtain this rather than using the scaler?
        num_times = len(set(self.unique_times_to_event)) - 1
        num_pat = len(weights) # Is there a nicer way to obtain this rather than usng the length of the weights
        times_to_fill = list(set(self.unique_times_to_event)); times_to_fill.sort()
        
        b = cp.Variable(num_features)

        # These next two lines are the implementation of the equation (21) of the paper
        cost = [weights[k]*cp.square(log_correction[k][j])*cp.square(cp.log(predictions[k][j]+epsilon) - cp.log(H0[j]+epsilon) - b @ neighbours[k])*(times_to_fill[j+1]-times_to_fill[j])\
                 for k in tqdm(range(num_pat)) for j in range(num_times)] # 
        
        print(f'time creating the cost list {timeit.default_timer() - start_time}')
        start_time = timeit.default_timer()

        cost_sum = cp.sum(cost)
        print(f'time summing the cost list {timeit.default_timer() - start_time}')
        start_time = timeit.default_timer()
        prob = cp.Problem(cp.Minimize(cost_sum))


        opt_val = prob.solve(verbose=True, max_iter=100000)
        print(f'time solving the problem {timeit.default_timer() - start_time}')
        return b.value

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
        self.sd_features = np.std(
            self.training_features, axis=0, dtype=self.training_features.dtype
        )
        self.sd_sq = np.square(self.sd_features)
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
        den = 2 * (self.kernel_width**2)
        weight = np.exp(num / den)
        return weight

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
