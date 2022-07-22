import pandas as pd
import numpy as np

veteran_path = '/home/carlos.hernandez/PhD/SurvLIME/survLime/datasets/veteran.csv'


def Loader(dataset_name: str = 'veterans') -> list([pd.DataFrame, np.ndarray]):
    df = pd.read_csv(veteran_path)
    df['status'] = [True if x == 1 else False for x in df['status']]
    df['y'] = [(x, y) for x, y in zip(df['status'], df['time'])]
    y = df.pop('y').to_numpy()
    x = df[['celltype', 'trt', 'karno', 'diagtime', 'age', 'prior']]
    return x, y


class RandomSurvivalData:
    def __init__(
        self,
        center: list[float],
        radius: float,
        coefficients: list[float],
        prob_event: float,
        lambda_weibull: float,
        v_weibull: float,
    ) -> None:
        if len(center) != len(coefficients):
            raise ValueError(
                'length of center and length of coefficients must be equal'
            )
        if lambda_weibull <= 0:
            raise ValueError('lambda_weibull must be greater than 0')
        if v_weibull <= 0:
            raise ValueError('v_weibull must be greater than 0')
        self.center = center
        self.radius = radius
        self.coefficients = coefficients
        self.prob_event = prob_event
        self.lambda_weibull = lambda_weibull
        self.v_weibull = v_weibull

    def spherical_data(self, num_points: int) -> np.ndarray:
        center = self.center
        radius = self.radius

        n_dim = len(center)
        zero_list = [0 for _ in range(n_dim)]

        # Uniform data in range [0, 1]
        u = np.random.uniform(low=0.0, high=1.0, size=(num_points, 1))
        uniform_rad_root = u ** (1 / n_dim)
        uniform_rad = radius * uniform_rad_root

        # Multivariate normal distribution
        I = np.identity(n_dim)
        X = np.random.multivariate_normal(mean=zero_list, cov=I, size=num_points)

        # Standarisation
        X_sqr = np.sum(np.square(X), axis=1, keepdims=True)
        norm_X = np.sqrt(X_sqr)
        X_unit = X / norm_X
        sphere_data = uniform_rad * X_unit
        X_location = sphere_data + center

        return X_location

    def survival_times(self, num_points: int, X: np.array) -> np.array:
        u = np.random.uniform(size=(num_points, 1))
        lamba_val = self.lambda_weibull
        v = self.v_weibull
        b = np.reshape(self.coefficients, newshape=(len(self.coefficients), 1))
        num = -np.log(u)
        den = lamba_val * np.exp(np.dot(X, b))
        T = (num / den) ** (1 / v)
        T = np.where(T > 2000, 2000, T)
        return T

    def random_event(self, num_points: int) -> np.array:
        prob_event = self.prob_event
        return np.where(np.random.uniform(size=num_points) <= prob_event, True, False)

    def random_survival_data(self, num_points: int) -> tuple:
        # Get spherical data
        X = self.spherical_data(num_points=num_points)

        # Get random survival time
        T = self.survival_times(num_points=num_points, X=X)

        # Get event variable
        delta = self.random_event(num_points=num_points)

        return (X, T, delta)
