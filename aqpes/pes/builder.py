import numpy as np
from typing import Dict, Any, Tuple, Optional

from aqpes.core.base import QuantumSystem, PES
from aqpes.pes.analytic import PES_REGISTRY


class PESBuilder(PES):
    def __init__(self, pes_type: str = "morse",
                 params: Optional[Dict[str, float]] = None,
                 system_name: str = "H2_1D",
                 mass: float = 1.0):
        system = QuantumSystem(name=system_name, mass=mass)
        super().__init__(name=f"{system_name}_{pes_type}", system=system)

        self.pes_type = pes_type
        self.params = params or {}
        self._impl = PES_REGISTRY[pes_type](self.params)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return self._impl.evaluate(x)

    def generate_grid(self, x_min: float, x_max: float,
                      n_points: int) -> Tuple[np.ndarray, np.ndarray]:
        grid = np.linspace(x_min, x_max, n_points)
        values = self.evaluate(grid)
        return grid, values

    def add_gaussian_noise(self, grid: np.ndarray, values: np.ndarray,
                           noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        noise = np.random.normal(0, noise_level * np.max(values), size=values.shape)
        return grid, values + noise
