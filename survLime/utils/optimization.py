import numpy as np
import cvxpy as cp
import cvxpy.atoms.affine.binary_operators.MulExpression as MulExp


class OptFuncionMaker:
    """TODO
    Given certain input parameters, this class
    returns the function to be optimized

    """

    def __init__(
        self,
        E: np.ndarray = None,
        weights: np.ndarray = None,
        log_correction: np.ndarray = None,
        delta_t: np.ndarray = None,
    ) -> None:

        self.E = E
        self.weights = weights
        self.log_correction = log_correction
        self.delta_t = delta_t

    def compute_function(self, norm: int = 2) -> MulExp:
        
        norm = 1
        if norm == 1:
            E_norm = cp.abs(self.E)
        elif norm == 'inf':
            E_norm = cp.norm(self.E, 'inf')
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
