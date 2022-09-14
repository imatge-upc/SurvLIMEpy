from typing import Union
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.binary_operators import MulExpression as MulExp


class OptFuncionMaker:
    """TODO
    Given certain input parameters, this class
    returns the function to be optimized

    """

    def __init__(
        self,
        E: np.ndarray,
        weights: np.ndarray,
        log_correction: np.ndarray,
        delta_t: np.ndarray,
    ) -> None:

        self.E = E
        self.weights = weights
        self.log_correction = log_correction
        self.delta_t = delta_t

    def compute_function(self, norm: Union[float, str] = 2) -> MulExp:

        if isinstance(norm, float) and norm < 1:
            raise ValueError(f"norm should be greater than 1, given value {norm}")
        if norm == 1:
            E_norm = cp.abs(self.E)
        elif norm == "inf":
            E_norm = cp.norm(self.E, "inf")
        elif norm % 2 != 0:
            E_abs = cp.abs(self.E)
            E_norm = cp.power(E_abs, norm)
        else:
            E_norm = cp.power(self.E, norm)

        V_sq = cp.square(self.log_correction)
        F = cp.multiply(E_norm, V_sq)
        G = F @ self.delta_t
        funct = G.T @ self.weights
        return funct
