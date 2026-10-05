"""简化的 H + H₂ 反应解析势能面 — 高斯垒模型 (反应路径 + 谐振近似)。

Eckart 势垒 + Harmonically coupled，适合教学演示。
势垒高度 ~0.4 eV (0.015 au)，符合 H+H₂ 反应的真实特征。
"""

import numpy as np
from typing import Dict, Any


class EckartBarrierPES:
    def __init__(self, params: Dict[str, float] = None):
        p = params or {}
        self.V0 = p.get("V0", 0.015)
        self.beta = p.get("beta", 1.5)
        self.k_r = p.get("k_r", 0.5)
        self.r0 = p.get("r0", 1.401)
        self.coupling = p.get("coupling", 0.08)

    def _ecker(self, x):
        A = np.asarray(x, dtype=float)
        return self.V0 / np.cosh(self.beta * A) ** 2

    def evaluate(self, R, r):
        R = np.asarray(R, dtype=float)
        r = np.asarray(r, dtype=float)
        V_reaction = self._ecker(R - 3.0)
        V_vibration = 0.5 * self.k_r * (r - self.r0) ** 2
        V_coupling = self.coupling * (r - self.r0) * np.tanh(self.beta * (R - 3.0))
        return V_reaction + V_vibration + V_coupling

    def __call__(self, R, r):
        return self.evaluate(R, r)


class EckartBuilder:
    def __init__(self, params: Dict[str, float] = None):
        self.params = params or {}
        self._pes = EckartBarrierPES(self.params)

    def evaluate_2d(self, R, r, theta=0.0):
        return self._pes.evaluate(R, r)

    def generate_grid(self, R_range, r_range, n_R=100, n_r=100):
        R_grid = np.linspace(*R_range, n_R)
        r_grid = np.linspace(*r_range, n_r)
        RR, rr = np.meshgrid(R_grid, r_grid, indexing="ij")
        V = self.evaluate_2d(RR, rr)
        return R_grid, r_grid, V
