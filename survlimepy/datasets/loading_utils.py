import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from typing import List


def prepare_splits(
    x: pd.DataFrame,
    y: np.ndarray,
    split_sizes: List = [0.1, 0.5],
    random_state: int = 1,
) -> List[List, List]:
    """Creates a random split with given sizes and random seed.

    Args:
        x (pd.DataFrame): variable containing the dataset.
        y (np.ndarray): variable containing the target values.
        split_sizes (List): variable containing the percentage of the first and second split.
        random_state (int): int: seed of the random state to be used when splitting.

    Returns:
        X_x: pd.DataFrame: three different DataFrames with the splits' data.
        y_y: ----: three different ---- with the splits' targets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        x.copy(),
        y,
        test_size=split_sizes[0],
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test.copy(), y_test, test_size=split_sizes[1], random_state=random_state
    )
    scaler = StandardScaler()

    X_train_processed = pd.DataFrame(
        data=scaler.fit_transform(X_train, y_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        data=scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    events_train = [x[0] for x in y_train]
    times_train = [x[1] for x in y_train]

    events_val = [x[0] for x in y_val]
    times_val = [x[1] for x in y_val]

    events_test = [x[0] for x in y_test]
    times_test = [x[1] for x in y_test]

    y_train = Surv.from_arrays(events_train, times_train)
    y_val = Surv.from_arrays(events_val, times_val)
    y_test = Surv.from_arrays(events_test, times_test)

    return X_train_processed, y_train, X_val, y_val, X_test_processed, y_test
