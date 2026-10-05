import numpy as np
from typing import Dict, Any


class BaseAnalyticPES:
    def __init__(self, params: Dict[str, float]):
        self.params = params

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)


class MorsePES(BaseAnalyticPES):
    def __init__(self, params: Dict[str, float]):
        super().__init__(params)
        self.D = params.get("D", 0.1744)
        self.alpha = params.get("alpha", 1.028)
        self.r0 = params.get("r0", 0.7416)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        r = np.asarray(x, dtype=float)
        return self.D * (1 - np.exp(-self.alpha * (r - self.r0))) ** 2


class HarmonicPES(BaseAnalyticPES):
    def __init__(self, params: Dict[str, float]):
        super().__init__(params)
        self.k = params.get("k", 1.0)
        self.r0 = params.get("r0", 0.7416)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        r = np.asarray(x, dtype=float)
        return 0.5 * self.k * (r - self.r0) ** 2


class LJ_PES(BaseAnalyticPES):
    def __init__(self, params: Dict[str, float]):
        super().__init__(params)
        self.epsilon = params.get("epsilon", 1.0)
        self.sigma = params.get("sigma", 1.0)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        r = np.asarray(x, dtype=float)
        sr = self.sigma / r
        return 4 * self.epsilon * (sr ** 12 - sr ** 6)


PES_REGISTRY = {
    "morse": MorsePES,
    "harmonic": HarmonicPES,
    "lj": LJ_PES,
}
