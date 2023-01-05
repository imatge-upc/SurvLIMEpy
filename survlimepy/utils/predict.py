import numpy as np
import pandas as pd
import copy
from sksurv.functions import StepFunction
from typing import Callable, Optional
from survlimepy.utils.step_function import transform_step_function


def interpolate_values(
    matrix: np.ndarray,
    unique_times_to_event: np.ndarray,
    model_output_times: np.ndarray,
) -> np.ndarray:
    """Transform an array of step functions to a matrix.

    Args:
        matrix (np.ndarray): matrix with values.
        unique_times_to_event (np.ndarray): unique times to event.
        model_output_times (np.ndarray): times considered by teh model.

    Returns:
        interpolated_matrix (np.ndarray): array containing the values.
    """
    interpolated_matrix = np.array(
        [
            np.interp(unique_times_to_event, model_output_times, matrix[i])
            for i in range(matrix.shape[0])
        ]
    )
    return interpolated_matrix


def validate_predicted_matrix(
    matrix: np.ndarray,
    individuals: Optional[np.ndarray] = None,
    expected_num_rows: Optional[int] = None,
    expected_num_cols: Optional[int] = None,
) -> None:
    """Validate the output.

    Args:
        matrix (np.ndarray): matrix to validate.
        individuals (Optional[np.ndarray]): individuals to be predicted.
        expected_num_rows (Optional[int]): expected number of rows.
        expected_num_cols (Optional[int]): expected number of columns.

    Returns:
        None.
    """
    total_rows = matrix.shape[0]
    total_cols = matrix.shape[1]
    are_nan_values = np.isnan(matrix).any()
    if are_nan_values:
        msg = "There are nan values produced by predict_fn function."
        if individuals is not None:
            idx = np.argwhere(np.isnan(matrix))
            example = str(individuals[idx[0][0], :])
            msg = f"{msg} Try to predict {example} array."
        raise ValueError(msg)
    if expected_num_rows:
        if total_rows != expected_num_rows:
            raise ValueError(
                f"The predicted function returns {total_rows} rows while expecting {expected_num_rows} rows."
            )
    if expected_num_cols:
        if total_cols != expected_num_cols:
            raise ValueError(
                f"The predicted function returns {total_cols} columns while expecting {expected_num_cols} columns."
            )

    return None


def predict_wrapper(
    predict_fn: Callable,
    data: np.ndarray,
    unique_times_to_event: np.ndarray,
    model_output_times: np.ndarray,
) -> np.ndarray:
    """Return the matrix with the values predicted.

    Args:
        predict_fn (Callable): function that computes cumulative hazard.
        data (np.ndarray): data to predict over.
        unique_times_to_event (np.ndarray): unique times to event.
        model_output_times (np.ndarray): output times of the bb model.

    Returns:
        predicted_values(np.ndarray): predicted values.
    """
    num_individuals = data.shape[0]
    number_unique_times = unique_times_to_event.shape[0]
    # Predict
    values_raw = predict_fn(data)
    # In case of a pd.DataFrame, force numpy
    if isinstance(values_raw, pd.DataFrame):
        values = values_raw.to_numpy()
    else:
        values = copy.deepcopy(values_raw)
    # If it is a numpy array
    if isinstance(values, np.ndarray):
        total_dim = values.ndim
        # If it is a matrix
        if total_dim == 2:
            total_cols = values.shape[1]
            # The produced number of columns matches with the number of unique times to event
            if total_cols == number_unique_times:
                predicted_values = values
            # The number of columns does not match with the number of unique times to event
            else:
                validate_predicted_matrix(
                    matrix=values,
                    individuals=data,
                    expected_num_rows=num_individuals,
                )
                predicted_values = interpolate_values(
                    matrix=values,
                    unique_times_to_event=unique_times_to_event,
                    model_output_times=model_output_times,
                )
        # It is an 1D array
        elif total_dim == 1:
            # It is a StepFunction
            if isinstance(values[0], StepFunction):
                predicted_values = transform_step_function(array_step_functions=values)
                if predicted_values.shape[1] != number_unique_times:
                    validate_predicted_matrix(
                        matrix=predicted_values,
                        individuals=data,
                        expected_num_rows=num_individuals,
                    )
                    predicted_values = interpolate_values(
                        matrix=predicted_values,
                        unique_times_to_event=unique_times_to_event,
                        model_output_times=model_output_times,
                    )
            else:
                raise NotImplemented("Unknown type of object.")
    else:
        raise TypeError("Unknown type of object.")

    validate_predicted_matrix(
        matrix=predicted_values,
        individuals=data,
        expected_num_rows=num_individuals,
        expected_num_cols=number_unique_times,
    )
    return predicted_values
