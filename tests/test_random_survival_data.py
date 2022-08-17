import pytest
from survLime.datasets.load_datasets import RandomSurvivalData
import random
import numpy as np

center = [0, 0]
radius = 1
coefficients = [1, 2]
prob_event = 0.9
lambda_weibull = 10 ** (-5)
v_weibull = 2


def test_value_error_center_coefficients() -> None:
    with pytest.raises(ValueError):
        rsd = RandomSurvivalData(
            center=[1, 2, 3],
            radius=radius,
            coefficients=coefficients,
            prob_event=prob_event,
            lambda_weibull=lambda_weibull,
            v_weibull=v_weibull,
        )


def test_value_error_negative_lambda_weibull() -> None:
    with pytest.raises(ValueError):
        rsd = RandomSurvivalData(
            center=center,
            radius=radius,
            coefficients=coefficients,
            prob_event=prob_event,
            lambda_weibull=-1,
            v_weibull=v_weibull,
        )


def test_value_error_negative_v_weibull() -> None:
    with pytest.raises(ValueError):
        rsd = RandomSurvivalData(
            center=center,
            radius=radius,
            coefficients=coefficients,
            prob_event=prob_event,
            lambda_weibull=lambda_weibull,
            v_weibull=-1,
        )


def test_shape_spherical_data() -> None:
    num_points = 100
    rsd = RandomSurvivalData(
        center=center,
        radius=radius,
        coefficients=coefficients,
        prob_event=prob_event,
        lambda_weibull=lambda_weibull,
        v_weibull=v_weibull,
    )
    data = rsd.spherical_data(num_points=num_points)
    data_shape = data.shape
    assert data_shape == (num_points, len(center))
