"""几何优化、数值 Hessian 与谐振频率 — 直接复用 Calculator 的解析梯度。

这补齐了 PES 工作流的前置环节: **先优化再扫描**。所有方法只依赖
``Calculator`` 协议 (能量 + 核梯度, Hartree/Bohr), 因此对解析面、
ML 代理面、以及任意电子结构后端 (SCF/DFT/MP2/CCSD(T)) 通用。

单位约定: 坐标 Bohr, 能量 Hartree, 梯度 Hartree/Bohr, 频率 cm⁻¹。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from aqpes.core.periodic import AMU_TO_ME, mass

#: 原子单位频率 → cm⁻¹ (ω[au] = sqrt(λ), λ 为质量加权 Hessian 本征值)
AU_FREQ_TO_CM = 219474.6313632


# ---------------------------------------------------------------------------
# 几何优化 (BFGS + Armijo 回溯线搜索)
# ---------------------------------------------------------------------------

def optimize_geometry(calc, coords0: np.ndarray, max_iter: int = 200,
                      gtol: float = 1e-5, step_max: float = 0.25,
                      verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """用解析梯度做 BFGS 极小化 (极小点几何)。

    Parameters
    ----------
    calc : Calculator
        任意满足协议的后端。
    coords0 : (N, 3) ndarray
        初始几何 (Bohr)。
    max_iter, gtol : int, float
        最大迭代数与梯度收敛阈值 (Hartree/Bohr)。
    step_max : float
        单步位移上限 (Bohr), 防止大步跳入非物理区。

    Returns
    -------
    coords : (N, 3) ndarray
        优化后几何 (Bohr)。
    info : dict
        ``converged / n_iter / energy / grad_max / n_evals``。
    """
    x = np.asarray(coords0, dtype=float).ravel().copy()
    n = x.size
    e, g = calc.energy_and_gradient(x.reshape(-1, 3))
    gf = np.asarray(g, dtype=float).ravel()
    Binv = np.eye(n)                 # 逆 Hessian 近似
    n_evals = 1
    converged = False
    for it in range(max_iter):
        gmax = float(np.abs(gf).max())
        if verbose and (it % 10 == 0 or gmax < gtol):
            print(f"  iter {it:3d}: E = {e:.8f} Ha, |g|max = {gmax:.2e}")
        if gmax < gtol:
            converged = True
            break
        p = -Binv @ gf
        pn = float(np.linalg.norm(p))
        if pn > step_max:
            p *= step_max / pn
        # Armijo 回溯
        alpha, accepted = 1.0, False
        for _ in range(40):
            xn = x + alpha * p
            en, gn = calc.energy_and_gradient(xn.reshape(-1, 3))
            n_evals += 1
            if en < e - 1e-4 * alpha * float(gf @ p):
                accepted = True
                break
            alpha *= 0.5
        if not accepted:            # 线搜索失败 → 已足够接近极小
            converged = True
            break
        s = xn - x
        y = np.asarray(gn, dtype=float).ravel() - gf
        sy = float(s @ y)
        if sy > 1e-10:              # BFGS 逆 Hessian 更新 (仅当曲率正定)
            I = np.eye(n)
            Binv = ((I - np.outer(s, y) / sy) @ Binv
                    @ (I - np.outer(y, s) / sy) + np.outer(s, s) / sy)
        x, e, gf = xn, en, np.asarray(gn, dtype=float).ravel()
    return x.reshape(-1, 3), {
        "converged": converged, "n_iter": it + 1, "energy": float(e),
        "grad_max": float(np.abs(gf).max()), "n_evals": n_evals,
    }


# ---------------------------------------------------------------------------
# 数值 Hessian 与谐振频率
# ---------------------------------------------------------------------------

def numerical_hessian(calc, coords: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """由解析梯度的中心差分构造 Hessian (Hartree/Bohr²)。

    成本: 6N 次梯度评估 (N = 3·n_atoms)。对称化以消除数值不对称。
    """
    c = np.asarray(coords, dtype=float)
    n = c.size
    H = np.zeros((n, n))
    for j in range(n):
        cp, cm = c.copy().ravel(), c.copy().ravel()
        cp[j] += h
        cm[j] -= h
        _, gp = calc.energy_and_gradient(cp.reshape(-1, 3))
        _, gm = calc.energy_and_gradient(cm.reshape(-1, 3))
        H[:, j] = (np.asarray(gp).ravel() - np.asarray(gm).ravel()) / (2 * h)
    return 0.5 * (H + H.T)


def harmonic_frequencies(calc, symbols: Sequence[str], coords: np.ndarray,
                         h: float = 1e-3, extra_masses: Optional[Dict] = None,
                         ) -> Tuple[np.ndarray, Dict]:
    """谐振频率 (cm⁻¹), 含平移/转动投影。

    Returns
    -------
    freqs : ndarray
        3N-6 (非线性) 或 3N-5 (线性) 个频率, 升序; 虚频以负数表示。
    info : dict
        ``hessian / n_imag / masses``。
    """
    c = np.asarray(coords, dtype=float)
    natm = c.shape[0]
    m = np.array([mass(s, extra_masses) for s in symbols]) * AMU_TO_ME
    H = numerical_hessian(calc, c, h=h)

    # 质量加权 + 去平动转动 (在笛卡尔坐标上用投影, 避免内坐标实现)
    sq = np.sqrt(np.repeat(m, 3))
    Hmw = H / np.outer(sq, sq)

    # 投影算子: 去掉平动 (3) 与转动 (3 或 2) 方向
    P = _transrot_basis(c, m)
    proj = np.eye(3 * natm) - P @ P.T
    Hmw = proj @ Hmw @ proj
    # ⚠ 必须在**平动/转动正交补空间内**对角化, 不能取"最大 nvib 个本征值":
    #   虚频对应**负**本征值, 会排到数值零模之下而被丢掉
    #   (H₃ 过渡态因此被误报为 0 虚频 — 真实踩过的 bug)。
    w, V = np.linalg.eigh(proj)
    Q = V[:, w > 0.5]                       # 振动子空间的正交基 (3N x nvib)
    Hvib = Q.T @ Hmw @ Q
    evals = np.linalg.eigvalsh(Hvib)
    freqs = np.sign(evals) * np.sqrt(np.abs(evals)) * AU_FREQ_TO_CM
    info = {"hessian": H, "n_imag": int((freqs < -1e-6).sum()), "masses": m}
    return np.sort(freqs), info


def _is_linear(coords: np.ndarray, tol: float = 1e-3) -> bool:
    """判断是否线性分子 (所有原子共线)。"""
    c = coords - coords.mean(axis=0)
    if c.shape[0] < 3:
        return True
    # 取最大主轴, 检查其余轴的残差
    _, s, Vt = np.linalg.svd(c)
    return bool(s[1] < tol * max(s[0], 1e-12))


def _transrot_basis(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """构造 (3N, k) 的平动+转动正交基 (质量加权, 已正交归一)。"""
    c = coords - coords.mean(axis=0)
    natm = c.shape[0]
    sq = np.sqrt(np.repeat(masses, 3))
    vecs = []
    # 平动
    for d in range(3):
        v = np.zeros((natm, 3))
        v[:, d] = 1.0
        vecs.append(v.ravel() * sq)
    # 转动 (绕主轴: x, y, z)
    for d in range(3):
        ax = np.zeros(3)
        ax[d] = 1.0
        v = np.cross(ax, c)
        vecs.append(v.ravel() * sq)
    M = np.array(vecs).T                       # (3N, 6)
    Q, R = np.linalg.qr(M)
    # 保留范数显著非零的列 (线性分子会有 5 个独立方向)
    keep = [i for i in range(Q.shape[1]) if abs(R[i, i]) > 1e-8]
    return Q[:, keep]
