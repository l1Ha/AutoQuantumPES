import numpy as np
from typing import Dict, Any, Optional

class QuantumSystem:
    """量子体系最小描述 (名称、质量与维数)。

    Attributes:
        name: 体系名称 (如 "H2_1D")。
        mass: 粒子质量 (电子质量单位)。
        n_dim: 自由度维数。
    """

    def __init__(self, name: str, mass: float, n_dim: int = 1):
        self.name = name
        self.mass = mass
        self.n_dim = n_dim

    def __repr__(self):
        return f"QuantumSystem(name={self.name}, mass={self.mass}, dim={self.n_dim})"


class PES:
    """势能面抽象基类: 子类须实现 ``evaluate`` 返回坐标处的势能。"""

    def __init__(self, name: str, system: QuantumSystem):
        self.name = name
        self.system = system

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """计算势能 (Hartree); 子类必须实现。"""
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)
