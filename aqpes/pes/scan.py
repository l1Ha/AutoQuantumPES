"""内坐标 PES 扫描驱动 — 从「笛卡尔位移网格」升级到「化学上有意义的扫描」。

提供三类扫描, 均产出可直接喂给 NN 拟合的 ``AbInitioData`` (含能量与核梯度):

- :func:`scan_bond`   — 拉伸/压缩指定原子对的键长 (刚性扫描, 其余坐标固定)
- :func:`scan_angle`  — 改变三原子键角
- :func:`scan_path`   — 在两个几何之间线性插值 (反应路径的零级近似)

刚性扫描 (rigid scan) 说明: 除被扫描的内坐标外其余原子保持不动。这对
键长/键角扫描是最常用的模式, 与 Gaussian 的 ``scan`` 默认行为一致;
松弛扫描 (relaxed scan) 需在每个点做约束优化, 属路线图项。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from aqpes.pes.abinitio import AbInitioData


def _dist(c: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(c[i] - c[j]))


def _angle(c: np.ndarray, i: int, j: int, k: int) -> float:
    """键角 j-i-k? 约定: 顶点为 j, 两臂指向 i 与 k (与化学惯例一致)。"""
    u = c[i] - c[j]
    v = c[k] - c[j]
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _set_bond(c: np.ndarray, i: int, j: int, r: float) -> np.ndarray:
    """把原子对 (i, j) 的距离设为 r, 沿原方向平移 i 端 (j 固定)。"""
    out = c.copy()
    d = c[i] - c[j]
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("原子对重合, 无法定义键方向")
    out[i] = c[j] + d / n * r
    return out


def _set_angle(c: np.ndarray, i: int, j: int, k: int, deg: float) -> np.ndarray:
    """把键角 (顶点 j, 臂 i–j–k) 设为 deg, 保持两臂长度与平面不变。"""
    out = c.copy()
    u = c[i] - c[j]
    v = c[k] - c[j]
    ru, rv = np.linalg.norm(u), np.linalg.norm(v)
    e1 = u / ru
    # 构造平面内正交基
    w = v - (v @ e1) * e1
    nw = np.linalg.norm(w)
    if nw < 1e-12:                       # 共线: 任取垂直方向
        e2 = np.cross(e1, [0.0, 0.0, 1.0])
        if np.linalg.norm(e2) < 1e-8:
            e2 = np.cross(e1, [0.0, 1.0, 0.0])
        e2 = e2 / np.linalg.norm(e2)
    else:
        e2 = w / nw
    a = np.radians(deg)
    out[k] = c[j] + rv * (np.cos(a) * e1 + np.sin(a) * e2)
    out[i] = c[j] + ru * e1              # i 端方向不变 (仅确保长度)
    return out


def _collect(calc, geoms: Sequence[np.ndarray], labels: Sequence[float],
             symbols: Sequence[str], coord_name: str) -> AbInitioData:
    """逐点算能量+梯度, 打包为 AbInitioData。"""
    n = len(geoms)
    energies = np.zeros(n)
    grads = np.zeros((n, geoms[0].shape[0], 3))
    for k, g in enumerate(geoms):
        e, gr = calc.energy_and_gradient(g)
        energies[k] = e
        grads[k] = gr
    data = AbInitioData()
    data.from_arrays(np.array([g.ravel() for g in geoms]), energies,
                     gradients=grads)
    data.symbols = list(symbols)                    # 供对称函数/委员会拟合
    data.geometry = np.array(geoms)                 # (n, N, 3) Bohr
    data.meta = {"scan": coord_name, "labels": np.asarray(labels),
                 "n_points": n}
    return data


def scan_bond(calc, symbols: Sequence[str], coords: np.ndarray,
              i: int, j: int, r_values: Sequence[float],
              ) -> AbInitioData:
    """刚性扫描键长 (i, j) (Bohr)。返回含能量/梯度的 AbInitioData。"""
    c0 = np.asarray(coords, dtype=float)
    geoms = [_set_bond(c0, i, j, float(r)) for r in r_values]
    return _collect(calc, geoms, list(r_values), symbols,
                    f"bond({i},{j})")


def scan_angle(calc, symbols: Sequence[str], coords: np.ndarray,
               i: int, j: int, k: int, angles_deg: Sequence[float],
               ) -> AbInitioData:
    """刚性扫描键角 i–j–k (顶点 j, 单位度)。"""
    c0 = np.asarray(coords, dtype=float)
    geoms = [_set_angle(c0, i, j, k, float(a)) for a in angles_deg]
    return _collect(calc, geoms, list(angles_deg), symbols,
                    f"angle({i},{j},{k})")


def scan_path(calc, symbols: Sequence[str], coords_a: np.ndarray,
              coords_b: np.ndarray, n_points: int = 9,
              ) -> AbInitioData:
    """两几何间线性插值路径 (Bohr, 0=起点, 1=终点)。"""
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    ts = np.linspace(0.0, 1.0, n_points)
    geoms = [(1 - t) * a + t * b for t in ts]
    data = _collect(calc, geoms, list(ts), symbols, "path")
    data.meta["endpoints"] = (a, b)
    return data


def relaxed_scan_bond(calc, symbols: Sequence[str], coords: np.ndarray,
                      i: int, j: int, r_values: Sequence[float],
                      gtol: float = 1e-4, **opt_kw) -> AbInitioData:
    """松弛键长扫描: 每个 R 点固定该键、优化其余自由度。

    实现: 在「固定键长约束」下做投影梯度 BFGS (把键方向分量从梯度中投影掉)。
    """
    from aqpes.pes.optimize import optimize_geometry

    c0 = np.asarray(coords, dtype=float)
    natm = c0.shape[0]
    geoms, labels = [], []
    for r in r_values:
        start = _set_bond(c0, i, j, float(r))

        def gproj(coords):        # 约束梯度: 去掉改变键长 i-j 的分量
            e, g = calc.energy_and_gradient(coords)
            n = coords[i] - coords[j]
            n = n / (np.linalg.norm(n) + 1e-30)
            gp = g.copy()
            gp[i] -= (g[i] @ n) * n
            gp[j] -= (g[j] @ n) * n
            return e, gp

        class _ProjCalc:
            def energy_and_gradient(self, coords):
                return gproj(np.asarray(coords, dtype=float))

        opt, info = optimize_geometry(_ProjCalc(), start, gtol=gtol, **opt_kw)
        opt = _set_bond(opt, i, j, float(r))     # 恢复精确键长
        geoms.append(opt); labels.append(float(r))
    return _collect(calc, geoms, labels, symbols, f"relaxed-bond({i},{j})")
