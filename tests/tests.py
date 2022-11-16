import numpy as np
from functools import partial
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.utils import check_random_state
from survlime import survlime_explainer
from survlime.utils.neighbours_generator import NeighboursGenerator
from survlime.datasets.load_datasets import Loader
from typing import List
import pandas as pd


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
    assert x.shape == (228, 8)


def test_shape_pbc_preprocessed() -> None:
    loader = Loader(dataset_name="pbc")
    x, _, _ = loader.load_data()
    assert x.shape == (419, 17)


def test_shape_vetearns_computed_weights() -> None:
    loader = Loader(dataset_name="veterans")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 9


def test_shape_udca_computed_weights() -> None:
    loader = Loader(dataset_name="udca")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 4


def test_shape_lung_computed_weights() -> None:
    loader = Loader(dataset_name="lung")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 11


def test_shape_pbc_computed_weights() -> None:
    loader = Loader(dataset_name="pbc")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    b = compute_weights(train, test)
    assert len(b) == 22


def test_norm_less_than_one() -> None:
    loader = Loader(dataset_name="veterans")
    x, events, times = loader.load_data()
    train, _, test = loader.preprocess_datasets(x, events, times, random_seed=0)
    try:
        _ = compute_weights(train, test, norm=0.5)
    except ValueError:
        pass


def test_categorical_features() -> None:

    n = 100
    random_state = check_random_state(2)
    data = {
        "col1": random_state.normal(size=n),
        "col2": random_state.normal(size=n),
        "col3": random_state.choice(a=["a", "b", "c"], size=n),
        "col4": random_state.choice(a=["d", "e", "f"], size=n),
    }

    df = pd.DataFrame(data)
    data_row = df.loc[0].to_numpy()

    neighbours_generator = NeighboursGenerator(
        training_data=df,
        data_row=data_row,
        categorical_features=[2, 3],
        random_state=random_state,
    )

    neighbours = neighbours_generator.generate_neighbours(100)
    neighbours_first = neighbours[0, 2:4]
    expected_results = neighbours_first[0] == "a" and neighbours_first[1] == "f"
    assert expected_results == True


def compute_weights(train: np.array, test: np.array, norm: float = 2) -> List[float]:
    model = CoxPHSurvivalAnalysis(alpha=0.0001)

    model.fit(train[0], train[1])

    times_to_fill = list(set([x[1] for x in train[1]]))
    times_to_fill.sort()

    explainer = survlime_explainer.SurvLimeExplainer(
        train[0], train[1], model_output_times=model.event_times_
    )

    num_pat = 1000
    test_point = test[0].iloc[0]
    predict_chf = partial(model.predict_cumulative_hazard_function, return_array=True)
    b, _ = explainer.explain_instance(
        test_point,
        predict_chf,
        verbose=False,
        num_samples=num_pat,
        norm=norm,
    )
    b = [x[0] for x in b]
    return b
