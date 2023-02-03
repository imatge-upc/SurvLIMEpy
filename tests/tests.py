import numpy as np
from functools import partial
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.utils import check_random_state
from survlimepy import SurvLimeExplainer
from survlimepy.load_datasets import RandomSurvivalData
from survlimepy.utils.neighbours_generator import NeighboursGenerator
from survlimepy.load_datasets import Loader
from typing import List, Dict
import pandas as pd
import pytest


def test_shape_veterans_preprocessed() -> None:
    loader = Loader(dataset_name="veterans")
    x, _, _ = loader.load_data()
    assert x.shape == (137, 6)


def test_shape_udca_preprocessed() -> None:
    loader = Loader(dataset_name="udca")
    x, _, _ = loader.load_data()
    assert x.shape == (170, 4)


def test_shape_lung_preprocessed() -> None:
    loader = Loader(dataset_name="lung")
    x, _, _ = loader.load_data()
    assert x.shape == (226, 8)


def test_shape_pbc_preprocessed() -> None:
    loader = Loader(dataset_name="pbc")
    x, _, _ = loader.load_data()
    assert x.shape == (419, 17)


def test_shape_vetearns_computed_weights() -> None:
    loader = Loader(dataset_name="veterans")
    x, events, times = loader.load_data()
    train, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 8


def test_shape_udca_computed_weights() -> None:
    loader = Loader(dataset_name="udca")
    x, events, times = loader.load_data()
    train, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 4


def test_shape_lung_computed_weights() -> None:
    loader = Loader(dataset_name="lung")
    x, events, times = loader.load_data()
    train, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 9


def test_norm_less_than_one() -> None:
    loader = Loader(dataset_name="veterans")
    x, events, times = loader.load_data()
    train, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    with pytest.raises(ValueError):
        compute_weights(train, test, norm=0.5)


def test_num_rows() -> None:
    loader = Loader(dataset_name="lung")
    x, events, times = loader.load_data()
    train, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    with pytest.raises(ValueError):
        compute_weights(train, test, mock_predict_fn=True)


def test_montecarlo_simulation() -> None:
    # Generate data
    n_points = 500
    true_coef = [1, 2]
    r = 1
    center = [0, 0]
    prob_event = 0.9
    lambda_weibull = 10 ** (-6)
    v_weibull = 2

    rsd = RandomSurvivalData(
        center=center,
        radius=r,
        coefficients=true_coef,
        prob_event=prob_event,
        lambda_weibull=lambda_weibull,
        v_weibull=v_weibull,
        time_cap=None,
        random_seed=90,
    )

    # Train
    X, time_to_event, delta = rsd.random_survival_data(num_points=n_points)
    z = [(d, t) for d, t in zip(delta, time_to_event)]
    y = np.array(z, dtype=[("delta", np.bool_), ("time_to_event", np.float32)])

    # Fit a Cox model
    cox = CoxPHSurvivalAnalysis()
    cox.fit(X, y)

    # User montecarlo
    data = np.array([[0, 0], [1, 1]])
    explainer = SurvLimeExplainer(
        training_features=X,
        training_events=[tp[0] for tp in y],
        training_times=[tp[1] for tp in y],
        model_output_times=cox.event_times_,
        random_state=10,
    )

    explanations = explainer.montecarlo_explanation(
        data=data,
        predict_fn=cox.predict_cumulative_hazard_function,
        num_samples=1,
        num_repetitions=1,
    )

    assert explanations.shape == data.shape


def test_value_error_center_coefficients(random_survidal_data) -> None:
    with pytest.raises(ValueError):
        RandomSurvivalData(
            center=[1, 2, 3],
            radius=random_survidal_data["radius"],
            coefficients=random_survidal_data["coefficients"],
            prob_event=random_survidal_data["prob_event"],
            lambda_weibull=random_survidal_data["lambda_weibull"],
            v_weibull=random_survidal_data["v_weibull"],
        )


def test_value_error_negative_lambda_weibull(random_survidal_data) -> None:
    with pytest.raises(ValueError):
        RandomSurvivalData(
            center=random_survidal_data["center"],
            radius=random_survidal_data["radius"],
            coefficients=random_survidal_data["coefficients"],
            prob_event=random_survidal_data["prob_event"],
            lambda_weibull=-1,
            v_weibull=random_survidal_data["v_weibull"],
        )


def test_value_error_negative_v_weibull(random_survidal_data) -> None:
    with pytest.raises(ValueError):
        RandomSurvivalData(
            center=random_survidal_data["center"],
            radius=random_survidal_data["radius"],
            coefficients=random_survidal_data["coefficients"],
            prob_event=random_survidal_data["prob_event"],
            lambda_weibull=random_survidal_data["lambda_weibull"],
            v_weibull=-1,
        )


def test_shape_spherical_data(random_survidal_data) -> None:
    num_points = 100
    center = random_survidal_data["center"]
    rsd = RandomSurvivalData(
        center=center,
        radius=random_survidal_data["radius"],
        coefficients=random_survidal_data["coefficients"],
        prob_event=random_survidal_data["prob_event"],
        lambda_weibull=random_survidal_data["lambda_weibull"],
        v_weibull=random_survidal_data["v_weibull"],
    )
    data = rsd.spherical_data(num_points=num_points)
    data_shape = data.shape
    assert data_shape == (num_points, len(center))


@pytest.fixture
def random_survidal_data() -> Dict:
    data = {
        "center": [0, 0],
        "radius": 1,
        "coefficients": [1, 2],
        "prob_event": 0.9,
        "lambda_weibull": 10 ** (-5),
        "v_weibull": 2,
    }
    return data


def compute_weights(
    train: np.array, test: np.array, norm: float = 2, mock_predict_fn: bool = False
) -> List[float]:
    model = CoxPHSurvivalAnalysis(alpha=0.0001)

    model.fit(train[0], train[1])
    events = [y[0] for y in train[1]]
    times = [y[1] for y in train[1]]

    times_to_fill = list(set([x[1] for x in train[1]]))
    times_to_fill.sort()

    explainer = SurvLimeExplainer(
        train[0],
        events,
        times,
        functional_norm=norm,
        model_output_times=model.event_times_,
    )

    num_pat = 1000
    test_point = test[0].iloc[0, :]

    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    if mock_predict_fn:
        predict_chf = predict_chf_mocked(model.predict_cumulative_hazard_function)

    b = explainer.explain_instance(
        test_point,
        predict_fn=predict_chf,
        verbose=False,
        num_samples=num_pat,
    )
    return b


def predict_chf_mocked(predict_fn):
    def inner(X):
        total_rows = X.shape[1]
        pred_values = predict_fn(X=X, return_array=True)
        return pred_values[: (total_rows - 1), :]

    return inner
