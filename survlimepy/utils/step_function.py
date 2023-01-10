import numpy as np


def transform_step_function(
    array_step_functions: np.array, return_column_vector: bool = False
) -> np.array:
    """Transform an array of step functions to a matrix.

    Args:
        array_step_functions (np.array): array of step functions.
        return_column_vector (bool): boolean indicating whether the value returned must be a column vector or not.

    Returns:
        matrix_values (np.array): array containing the values.
    """
    sample_value = array_step_functions[0]
    total_rows = len(array_step_functions)
    total_columns = len(sample_value.y)
    matrix_values = np.empty(shape=(total_rows, total_columns))
    for i, step_fn in enumerate(array_step_functions):
        matrix_values[i] = step_fn(step_fn.x)
    if return_column_vector:
        n_rows, n_cols = matrix_values.shape
        if n_rows == 1 and n_cols >= 1:
            matrix_values = matrix_values.T
    return matrix_values
