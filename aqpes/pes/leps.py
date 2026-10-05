"""LEPS (London-Eyring-Polanyi-Sato) 势能面 — H + H₂ 反应的标准模型。

参考文献:
  - Sato, S. J. Chem. Phys. 23, 592 (1955)
  - Schatz, G. C. Rev. Mod. Phys. 61, 669 (1989)
"""

import numpy as np
from typing import Dict, Any


class LEPSPES:
    def __init__(self, params: Dict[str, float]):
        self.D = params.get("D", 0.1744)
        self.alpha = params.get("alpha", 1.028)
        self.r0 = params.get("r0", 1.401)
        self.sato = params.get("sato", 0.05)

    def _singlet(self, r):
        x = np.asarray(r, dtype=float)
        return self.D * (np.exp(-2 * self.alpha * (x - self.r0))
                         - 2 * np.exp(-self.alpha * (x - self.r0)))

    def _triplet(self, r):
        x = np.asarray(r, dtype=float)
        return self.D * (np.exp(-2 * self.alpha * (x - self.r0))
                         + 2 * np.exp(-self.alpha * (x - self.r0)))

    def evaluate(self, r_AB, r_BC, r_AC):
        r_AB = np.asarray(r_AB, dtype=float)
        r_BC = np.asarray(r_BC, dtype=float)
        r_AC = np.asarray(r_AC, dtype=float)

        Q_AB = 0.5 * (self._singlet(r_AB) + self._triplet(r_AB))
        Q_BC = 0.5 * (self._singlet(r_BC) + self._triplet(r_BC))
        Q_AC = 0.5 * (self._singlet(r_AC) + self._triplet(r_AC))

        J_AB = 0.5 * (self._singlet(r_AB) - self._triplet(r_AB))
        J_BC = 0.5 * (self._singlet(r_BC) - self._triplet(r_BC))
        J_AC = 0.5 * (self._singlet(r_AC) - self._triplet(r_AC))

        s = self.sato
        J_AB *= (1.0 + s) / (1.0 - s)
        J_BC *= (1.0 + s) / (1.0 - s)
        J_AC *= (1.0 + s) / (1.0 - s)

        Q = Q_AB + Q_BC + Q_AC
        J2 = ((J_AB - J_BC) ** 2 + (J_BC - J_AC) ** 2 + (J_AC - J_AB) ** 2)

        V = Q - np.sqrt(0.5 * np.maximum(J2, 0))
        return V

    def __call__(self, r_AB, r_BC, r_AC=None):
        if r_AC is None:
            r_AC = r_AB + r_BC
        return self.evaluate(r_AB, r_BC, r_AC)


class LEPSBuilder:
    def __init__(self, params: Dict[str, float] = None):
        self.params = params or {}
        self._pes = LEPSPES(self.params)

    def evaluate_2d(self, R, r, theta=0.0):
        R = np.asarray(R, dtype=float)
        r = np.asarray(r, dtype=float)
        r_AB = r
        r_BC = R
        r_AC = r + R
        return self._pes.evaluate(r_AB, r_BC, r_AC)

    def generate_grid(self, R_range, r_range, n_R=100, n_r=100):
        R_grid = np.linspace(*R_range, n_R)
        r_grid = np.linspace(*r_range, n_r)
        RR, rr = np.meshgrid(R_grid, r_grid, indexing="ij")
        V = self.evaluate_2d(RR, rr)
        return R_grid, r_grid, V
