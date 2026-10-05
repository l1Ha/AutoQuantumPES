"""置换/平移/旋转对称函数 (Behler-Parrinello 型) 与共享原子能量架构。

为什么需要: 笛卡尔坐标特征的 NN 势能面**没有**对称性 — 交换两个
同种原子会得到不同的网络输入, 网络必须从数据里重新学习"交换后能量
不变"。本模块给出两条正路:

1. **对称函数描述符** (本文件): 只用距离与夹角构造逐原子中心的特征,
   严格满足置换/平移/旋转不变;
2. **共享原子能量架构** (``nn/ensemble.py``): 单个原子网络作用于每个
   中心的特征, 总能量 = Σ_i E_atom(Φ_i) — 原子置换只置换求和项,
   总能量**按构造严格不变**, 且同种原子共享参数 (样本效率提升)。

特征 (原子 i, 中心元素 A):

    G_i^A = Σ_{j≠i} f_c(r_ij)·exp(−η[(r_ij−r_c)²])

    G_i^{A,ζ} = Σ_{j<k; j,k≠i} 2^{1−ζ}(1+λ cos θ_jik)
                ·f_c(r_ij) f_c(r_ik)
                ·exp(−η[((r_ij+r_ik)/2−r_c)²])

f_c 为 Behler 平滑截断。角度项为 O(N²) — 本实现面向 N≲20 的教学/研究
原型; 生产级实现应配合邻居表与 GPU。

参考文献: J. Behler, M. Parrinello, PRL 98, 025901 (2007).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SymmetryFunctionParams:
    """对称函数超参数 (数据类)。

    Attributes:
        r_cut: Behler 平滑截断半径 (Bohr)。
        radial_etas: 径向高斯环宽度参数 η 序列 (需与键长尺度匹配)。
        angular_zetas: 角度项阶数 ζ 序列 (>= 1 的整数)。
        lam: 角度项余弦系数 λ (±1)。
        include_angular: 是否包含角度特征 (O(N²) 开销)。
    """
    # 径向项 exp(-η[(r-r_c)²]) 是以 r_c 为中心的高斯环: η 需与
    # (r_c - r_bond)² 匹配, 否则小分子键长 (1-3 Bohr) 全部落在
    # 高斯尾部 → 死特征 (实测 η≥4, r_c=5 时特征 ~1e-10)
    r_cut: float = 5.0
    radial_etas: Tuple[float, ...] = (0.02, 0.2, 1.0)
    angular_zetas: Tuple[int, ...] = (1, 2)
    lam: float = 1.0
    include_angular: bool = True

    def validate(self):
        """校验参数合法性, 非法时抛出 ValueError。"""
        if self.r_cut <= 0:
            raise ValueError("r_cut 必须为正")
        if not self.radial_etas:
            raise ValueError("至少需要一个径向 η")
        if any(int(z) < 1 for z in self.angular_zetas):
            raise ValueError("ζ 必须为 >=1 的整数阶")


def _cutoff(r: np.ndarray, r_cut: float) -> np.ndarray:
    """Behler 平滑截断 f_c(r) = ½[cos(πr/r_c)+1] (r<r_c), 否则 0。"""
    out = np.zeros_like(r)
    mask = r < r_cut
    out[mask] = 0.5 * (np.cos(np.pi * r[mask] / r_cut) + 1.0)
    return out


class SymmetryFunctionSet:
    """对称函数集: 几何 (n, N, 3) → 逐原子中心特征 (n, N, F)。

    特征布局对所有中心一致 (按全局元素表排列), 因此任意原子置换只
    置换中心维, 与共享原子能量求和架构组合即得严格置换不变能量。
    """

    def __init__(self, symbols: Sequence[str],
                 params: Optional[SymmetryFunctionParams] = None):
        params = params or SymmetryFunctionParams()
        params.validate()
        self.symbols = list(symbols)
        self.params = params
        self.species = sorted(set(self.symbols))
        self._index = {s: i for i, s in enumerate(self.symbols)}
        # 显式枚举中心特征的 (种类, 通道) 布局 — 计数与计算共用
        self._layout = self._build_layout()
        self.n_features = len(self._layout)
        self.n_centers = len(self.symbols)

    def _build_layout(self) -> List[tuple]:
        p = self.params
        layout: List[tuple] = []
        m = len(self.species)
        for s in self.species:                      # 径向: 邻居元素 × η
            for eta in p.radial_etas:
                layout.append(("radial", s, float(eta)))
        if p.include_angular:
            counts = {s: self.symbols.count(s) for s in self.species}
            for a in range(m):
                sa = self.species[a]
                for b in range(a, m):
                    sb = self.species[b]
                    n_pairs = (counts[sa] * (counts[sa] - 1) // 2 if a == b
                               else counts[sa] * counts[sb])
                    for z in p.angular_zetas:
                        layout.append(("angular", sa, sb, int(z), n_pairs))
        return layout

    def compute(self, coords: np.ndarray) -> np.ndarray:
        """逐原子中心对称函数。

        Parameters
        ----------
        coords : (n, N, 3) 或 (N, 3) 原子坐标 (Bohr)

        Returns
        -------
        (n, N, F) 特征张量 — 平移/旋转不变; 原子置换只置换中心维。
        """
        c = np.asarray(coords, dtype=float)
        if c.ndim == 2:
            c = c[None]
        if c.ndim != 3 or c.shape[1] != len(self.symbols):
            raise ValueError(
                f"coords 形状应为 (n, {len(self.symbols)}, 3), 收到 {c.shape}")
        if not np.all(np.isfinite(c)):
            raise ValueError("坐标含非有限值")
        n, N, _ = c.shape
        p = self.params

        d = c[:, :, None, :] - c[:, None, :, :]             # (n,N,N,3)
        r = np.linalg.norm(d, axis=-1)                        # (n,N,N)
        r_safe = np.maximum(r, 1e-10)
        u = d / r_safe[..., None]
        eye = np.eye(N, dtype=bool)[None, :, :]
        r = np.where(eye, np.inf, r)
        fc = _cutoff(r, p.r_cut)

        feats = np.zeros((n, N, self.n_features))
        eta_a = p.radial_etas[0]
        m = len(self.species)
        for i in range(N):
            ri = r[:, i, :]          # (n, N)
            fui = fc[:, i, :]
            ui = u[:, i, :]          # (n, N, 3)
            col = 0
            # --- 径向: 邻居元素 × η (无该元素邻居则置 0) ---
            for s in self.species:
                js = [j for j in range(N) if self.symbols[j] == s and j != i]
                rr = ri[:, js] if js else None
                fcc = fui[:, js] if js else None
                for eta in p.radial_etas:
                    feats[:, i, col] = (
                        np.sum(fcc * np.exp(-eta * (rr - p.r_cut) ** 2),
                               axis=1) if js else 0.0)
                    col += 1
            # --- 角度: 元素对 (a≤b) × ζ ---
            if p.include_angular:
                for a in range(m):
                    sa = self.species[a]
                    for b in range(a, m):
                        sb = self.species[b]
                        for zeta in p.angular_zetas:
                            feats[:, i, col] = self._angular_value(
                                ri, fui, ui, i, sa, sb, zeta, eta_a, p)
                            col += 1
            if col != self.n_features:
                raise RuntimeError(
                    f"特征计算与布局不一致: 中心 {i} 写入 {col} 列, "
                    f"布局 {self.n_features} 列")
        return feats

    def _angular_value(self, ri, fui, ui, i, sa, sb, zeta, eta_a, p) -> np.ndarray:
        js = [j for j in range(self.n_centers)
              if self.symbols[j] == sa and j != i]
        ks = [k for k in range(self.n_centers)
              if self.symbols[k] == sb and k != i]
        if not js or not ks:
            return np.zeros(ri.shape[0])
        pref = 2.0 ** (1 - zeta)
        total = np.zeros(ri.shape[0])
        for j in js:
            for k in ks:
                if sa == sb and k <= j:   # 同元素对只取 j<k, 避免双计
                    continue
                cos = np.sum(ui[:, j] * ui[:, k], axis=-1)
                rmean = (ri[:, j] + ri[:, k]) / 2.0
                total += (pref * (1.0 + p.lam * cos)
                          * fui[:, j] * fui[:, k]
                          * np.exp(-eta_a * (rmean - p.r_cut) ** 2))
        return total

    def compute_flat(self, coords: np.ndarray) -> np.ndarray:
        """展平特征 (n, N*F) — 供普通 MLP 使用。

        注意: 普通 MLP 拼接展平特征**不保证**置换不变; 需要严格对称性
        时请用共享原子能量架构 (nn/ensemble.AtomicEnergyCommittee)。
        """
        f = self.compute(coords)
        n = f.shape[0]
        return f.reshape(n, -1)
