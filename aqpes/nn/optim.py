"""Adam 优化器 (NumPy) — 供训练器与对称函数训练共用。"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


class Adam:
    """Adam (Kingma & Ba, 2015)。对参数列表原地更新。"""

    def __init__(self, shapes: Sequence[Sequence[int]], lr: float = 0.005,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros(s, dtype=float) for s in shapes]
        self.v = [np.zeros(s, dtype=float) for s in shapes]

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> None:
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
