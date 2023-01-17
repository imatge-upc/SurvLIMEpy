from typing import List, Optional
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv


data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets")
veteran_path = os.path.join(data_path, "veteran.csv")
udca_path = os.path.join(data_path, "udca_dataset.csv")
pbc_path = os.path.join(data_path, "pbc_dataset.csv")
lung_path = os.path.join(data_path, "lung_dataset.csv")
heart_path = os.path.join(data_path, "heart_dataset.csv")


class Loader:
    def __init__(self, dataset_name: str = "veterans"):

        if dataset_name == "veterans":
            self.feature_columns = [
                "celltype",
                "trt",
                "karno",
                "diagtime",
                "age",
                "prior",
            ]
            self.categorical_columns = ["celltype"]
            self.df = pd.read_csv(veteran_path)
            # if value in prior is different from 0 change to 1
            self.df["prior"] = [1 if x != 0 else 0 for x in self.df["prior"]]
            # substract 1 from variable trt
            self.df["trt"] = [x - 1 for x in self.df["trt"]]
        elif dataset_name == "udca":
            self.feature_columns = ["bili", "stage", "riskscore", "trt"]
            self.categorical_columns = []
            self.df = pd.read_csv(udca_path)
        elif dataset_name == "pbc":
            self.feature_columns = [
                "age",
                "bili",
                "chol",
                "albumin",
                "ast",
                "ascites",
                "copper",
                "alk.phos",
                "trig",
                "platelet",
                "protime",
                "trt",
                "sex",
                "hepato",
                "spiders",
                "edema",
                "stage",
            ]
            self.categorical_columns = ["edema", "stage"]
            self.df = pd.read_csv(pbc_path)
            self.df["sex"] = [1 if x == "f" else 0 for x in self.df["sex"]]
        elif dataset_name == "lung":
            self.feature_columns = [
                "inst",
                "age",
                "sex",
                "ph.ecog",
                "ph.karno",
                "pat.karno",
                "meal.cal",
                "wt.loss",
            ]
            self.categorical_columns = ["ph.ecog"]
            self.df = pd.read_csv(lung_path)
            # Delete the row with ph.ecog = 3.0
            self.df = self.df[self.df["ph.ecog"] != 3.0]
            # delete the row with ph.ecog = nan
            self.df = self.df[self.df["ph.ecog"].notna()]
            # substract 1 to each value of the status column
            self.df["status"] = [x - 1 for x in self.df["status"]]

        elif dataset_name == "synthetic":
            ## TODO
            pass
        elif dataset_name == "heart":
            self.feature_columns = [
                "age",
                "anaemia",
                "creatinine_phosphokinase",
                "diabetes",
                "ejection_fraction",
                "high_blood_pressure",
                "platelets",
                "serum_creatinine",
                "serum_sodium",
                "sex",
                "smoking",
            ]
            self.categorical_columns = []
            self.df = pd.read_csv(heart_path)
        else:
            raise AssertionError(
                f"The give name {dataset_name} was not found in [veterans, udca, pbc, lung]."
            )

    def load_data(self) -> list([pd.DataFrame, np.ndarray]):
        """
        Loads a survival dataset.

        Returns:
        x (pd.DataFrame): unprocessed features.
        y (np.ndarray): tuples with (status, time).
        """
        self.df["status"] = [True if x == 1 else False for x in self.df["status"]]
        self.df["y"] = [
            (x, y) for x, y in zip(self.df["status"], self.df["time"])
        ]  # Needed for sksurv
        y = self.df.pop("y").to_numpy()
        events = [x[0] for x in y]
        times = [x[1] for x in y]
        x = self.df[self.feature_columns]

        x = x.fillna(value=x.median(numeric_only=True))

        return x, events, times

    def preprocess_datasets(
        self,
        x: pd.DataFrame,
        events: list,
        times: list,
        standarize: bool = True,
        random_seed: int = 0,
    ) -> list([pd.DataFrame, np.ndarray]):
        """
        Preprocesses the data to be used as model input.

        For now it only converts categorical features to OHE and
        standarizes the data.
        """
        # Deal with categorical features
        x_pre = x.copy()
        for cat_feat in self.categorical_columns:
            names = [cat_feat + "_" + str(value) for value in x_pre[cat_feat].unique()]
            df_dummy_i = pd.get_dummies(
                x_pre[cat_feat], drop_first=True, prefix=cat_feat
            )
            new_names = df_dummy_i.columns
            # x_pre[names[1:]] = pd.get_dummies(x_pre[cat_feat], drop_first=True)
            x_pre[new_names] = df_dummy_i
            x_pre.drop(cat_feat, inplace=True, axis=1)

        # Then convert the data and the features to three splits
        # and standarize them
        y = Surv.from_arrays(events, times)
        X_train, X_test, y_train, y_test = train_test_split(
            x_pre.copy(), y, test_size=0.10, random_state=random_seed
        )
        # X_val, X_test, y_val, y_test = train_test_split(
        #    X_test.copy(), y_test, test_size=0.5, random_state=random_seed
        # )

        if standarize:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                data=scaler.fit_transform(X_train, y_train),
                columns=X_train.columns,
                index=X_train.index,
            )

            #    X_val = pd.DataFrame(
            #        data=scaler.transform(X_val),
            #        columns=X_val.columns,
            #        index=X_val.index,
            #    )

            X_test = pd.DataFrame(
                data=scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

        return [X_train, y_train], [X_test, y_test]  # [X_val, y_val]


class RandomSurvivalData:
    """Generate spherical random survival data."""

    def __init__(
        self,
        center: List[float],
        radius: float,
        coefficients: List[float],
        prob_event: float,
        lambda_weibull: float,
        v_weibull: float,
        time_cap: float = 2000,
        random_seed: Optional[int] = None,
    ) -> None:
        """Init.
        Args:
            center (float): center of the sphere.
            radius (float): radius of the sphere.
            coefficients (list): value of the covariates.
            prob_event (float): probability of an event occurring.
            lambda_weibull (float): scale parameter of a Weibull distribution.
            v_weibull (float): shape parameter of a Weibull distribution.
            time_cap (float): if the time is greater than time_cap, then time_cap will be used
            random_seed (Optional[int]):  number to be used for random seeds.

        Returns:
            None.
        """
        if len(center) != len(coefficients):
            raise ValueError(
                "length of center and length of coefficients must be equal."
            )
        if prob_event <= 0:
            raise ValueError("prob_event must be greater than 0.")
        if prob_event >= 1:
            raise ValueError("prob_event must be less than 0.")
        if lambda_weibull <= 0:
            raise ValueError("lambda_weibull must be greater than 0.")
        if v_weibull <= 0:
            raise ValueError("v_weibull must be greater than 0.")
        self.center = center
        self.radius = radius
        self.coefficients = coefficients
        self.prob_event = prob_event
        self.lambda_weibull = lambda_weibull
        self.v_weibull = v_weibull
        self.random_state = np.random.default_rng(random_seed)
        self.time_cap = time_cap

    def spherical_data(self, num_points: int) -> np.ndarray:
        """Generates random data in the p-dimensional sphere (covariates).

        Args:
            num_points (int): number of individuals to generate.

        Returns:
            np.ndarray: matrix with num_points rows and p columns, where p is the dimension of the space.
        """
        center = self.center
        radius = self.radius

        n_dim = len(center)
        zero_list = [0 for _ in range(n_dim)]

        # Uniform data in range [0, 1]
        u = self.random_state.uniform(low=0.0, high=1.0, size=(num_points, 1))
        uniform_rad_root = u ** (1 / n_dim)
        uniform_rad = radius * uniform_rad_root

        # Multivariate normal distribution
        I = np.identity(n_dim)
        X = self.random_state.multivariate_normal(
            mean=zero_list, cov=I, size=num_points
        )

        # Standarisation
        X_sqr = np.sum(np.square(X), axis=1, keepdims=True)
        norm_X = np.sqrt(X_sqr)
        X_unit = X / norm_X
        sphere_data = uniform_rad * X_unit
        X_location = sphere_data + center

        return X_location

    def survival_times(self, num_points: int, X: np.ndarray) -> np.ndarray:
        """Generates survival times following a Weibull distribution.

        Args:
            num_points (int):  number of individuals to generate.
            X (np.ndarray): matrix with num_points rows and p columns, where p is the dimension of the space.

        Returns:
            np.ndarray: a column vector containing the survival times.
        """
        u = self.random_state.uniform(size=(num_points, 1))
        lamba_val = self.lambda_weibull
        v = self.v_weibull
        b = np.reshape(self.coefficients, newshape=(len(self.coefficients), 1))
        num = -np.log(u)
        den = lamba_val * np.exp(np.dot(X, b))
        # Use a Weibull distribution
        time_to_event = (num / den) ** (1 / v)
        if self.time_cap:
            time_to_event = np.where(
                time_to_event > self.time_cap, self.time_cap, time_to_event
            )
        if len(time_to_event.shape) == 2:
            time_to_event = time_to_event[:, 0]
        return time_to_event

    def random_event(self, num_points: int) -> np.ndarray:
        """Generates random events following a binomial distributiom with probabilty `prob_event`.

        Args:
            num_points (int):  number of individuals to generate.

        Returns:
            np.ndarray: a column vector containing the random events.
        """
        prob_event = self.prob_event
        return np.where(
            self.random_state.uniform(size=num_points) <= prob_event, True, False
        )

    def random_survival_data(self, num_points: int) -> tuple:
        """Generates random survival data.

        Args:
            num_points (int):  number of individuals to generate.

        Returns:
            tuple: (X, time_to_event, delta), where:
                - X: matrix with num_points rows and p columns, where p is the dimension of the space.
                - time_to_event: a column vector containing the survival times.
                - delta: a column vector containing the random events.
        """
        # Get spherical data
        X = self.spherical_data(num_points=num_points)

        # Get random survival time
        time_to_event = self.survival_times(num_points=num_points, X=X)

        # Get event variable
        delta = self.random_event(num_points=num_points)

        return (X, time_to_event, delta)
