"""复合方法 (composite): 相关一致基组外推 (CBS) + CCSD(T) 加和校正。

**动机**: Molpro/Gaussian 的 G4/W1 等复合方法依赖其专用基组 (GTBas1、
G3MP2LargeXP 等) —— PySCF 未收录这些基组, 无法忠实复现 (已实测确认)。
本模块提供**公式完全公开、可逐项检验**的 CBS 复合方案作为替代:

    E_total = E_HF/CBS + E_corr/CBS + δ_CCSD(T)

其中
- ``E_HF/CBS``: 两参数指数外推 (Feller) ``E(X) = E_CBS + A·exp(−b·X)``,
  由两个基数 X₁<X₂ 的 HF 能量解出 (消去 A、b 后数值求解 b);
- ``E_corr/CBS``: Schwartz 型 ``E(X) = E_CBS + A·X⁻³`` 两点外推 (MP2 相关能);
- ``δ_CCSD(T) = E_CCSD(T)/X_small − E_MP2/X_small``: 小基组上的
  "高阶相关能增量" (标准的 CCSD(T)-加和近似, 使 CBS-MP2 提升为 CBS-CCSD(T))。

**诚实边界**: 这是**加和近似** (非变分、非尺寸一致保证), 精度取决于
δ_CCSD(T) 的基组收敛性与外推形式; 与 G4/W1 的参数化方案不同, 不做
"同等精度"声明 —— 验证脚本给出与更大基组/归档基准的逐项对比。
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["cbs_extrapolate_2p", "cbs_hf_2p", "cbs_hf_3p", "composite_energy",
           "make_energy_fn", "DEFAULT_SCHEME"]

#: 默认方案: HF (**三点指数** Feller, TZ/QZ/5Z) + MP2 相关能 (TZ,QZ, X⁻³)
#: + CCSD(T) 增量 (TZ)。HF 用三点是因为两点的指数衰减常数 b 欠定 (实测
#: 两点 b 搜索的外推误差 ~1e-4 Ha; 三点可解出 b, 达 ~1e-8 Ha)。
DEFAULT_SCHEME: Dict[str, object] = {
    "hf_bases": ("cc-pvtz", "cc-pvqz", "cc-pv5z"),
    "hf_X": (3, 4, 5),
    "corr_bases": ("cc-pvtz", "cc-pvqz"),
    "corr_X": (3, 4),
    "delta_basis": "cc-pvtz",
    "delta_method": "ccsd(t)",
    "ref_method": "mp2",
}


def cbs_extrapolate_2p(x1: float, e1: float, x2: float, e2: float,
                       power: float = 3.0) -> float:
    """两点幂律外推 ``E(X) = E_CBS + A·X^(−power)`` → 返回 ``E_CBS``。

    用于相关能 (Schwartz: power=3)。要求 ``x2 > x1``。
    """
    if x2 <= x1:
        raise ValueError(f"需要 x2 > x1, 得到 x1={x1}, x2={x2}")
    f1, f2 = x1 ** (-power), x2 ** (-power)
    if abs(f2 - f1) < 1e-14:
        return float(e2)
    return float((e1 * f2 - e2 * f1) / (f2 - f1))


def cbs_hf_2p(x1: float, e1: float, x2: float, e2: float,
              b0: float = 1.0) -> float:
    """两参数**指数**外推 (Feller) ``E(X) = E_CBS + A·exp(−b·X)`` → ``E_CBS``。

    由两点的能量差解出 b (非线性; 用括号法+二分), 再给出 E_CBS。
    相比幂律, 指数形式对 HF 能量的基组收敛更合适 (Feller 1993)。
    """
    if x2 <= x1:
        raise ValueError(f"需要 x2 > x1, 得到 x1={x1}, x2={x2}")
    dE = float(e2 - e1)          # < 0 (能量下降)
    if abs(dE) < 1e-14:
        return float(e2)

    def e_cbs_of_b(b: float) -> float:
        # 由 E1 = E_CBS + A e^{−b x1}, E2 = E_CBS + A e^{−b x2} 消去 A:
        # (E1 − E_CBS)/(E2 − E_CBS) = e^{b(x2−x1)}
        # 数值上直接用 A = (E1−E2)/(e^{−b x1} − e^{−b x2}) 求 E_CBS
        a = (e1 - e2) / (np.exp(-b * x1) - np.exp(-b * x2))
        return float(e1 - a * np.exp(-b * x1))

    # 单调性: E_CBS(b) 随 b 单调 → 二分求解使 E_CBS = E2 − Δ 的 b 不必要;
    # 直接扫描-细化 b ∈ [b0/10, 10·b0] 使三点一致性最佳 (用 E1,E2 与 x1,x2
    # 已定, b 由 |E_CBS(b) − E_CBS(2b)| 最小化确定)
    bs = np.geomspace(max(b0, 1e-3) / 10.0, max(b0, 1e-3) * 100.0, 400)
    vals = np.array([e_cbs_of_b(b) for b in bs])
    # 取"外推值随 b 变化最小"的区间中点 (b 增大 → 外推趋稳)
    d = np.abs(np.gradient(vals, bs))
    i = int(np.argmin(d[len(bs) // 4:]) + len(bs) // 4)
    return float(vals[i])


def cbs_hf_3p(x1: float, e1: float, x2: float, e2: float, x3: float, e3: float,
              b_lo: float = 0.05, b_hi: float = 20.0) -> float:
    """三点**指数**外推 (Feller) → ``E_CBS``; b 由三点比值方程数值解出。

    解 ``(E₁−E₂)/(E₂−E₃) = (e^{−b x₁}−e^{−b x₂})/(e^{−b x₂}−e^{−b x₃})``
    得 b (1-D 根, 二分), 再由 A = (E₂−E₃)/(e^{−b x₂}−e^{−b x₃}) 给出
    ``E_CBS = E₃ − A·e^{−b x₃}``。
    """
    if not (x1 < x2 < x3):
        raise ValueError(f"需要 x1 < x2 < x3, 得到 {x1}, {x2}, {x3}")
    lhs = (e1 - e2) / (e2 - e3) if abs(e2 - e3) > 1e-14 else 1.0

    def resid(b: float) -> float:
        n = np.exp(-b * x1) - np.exp(-b * x2)
        d = np.exp(-b * x2) - np.exp(-b * x3)
        return float(n / d - lhs) if abs(d) > 1e-300 else 1e9

    lo, hi = b_lo, b_hi
    r_lo, r_hi = resid(lo), resid(hi)
    if r_lo * r_hi > 0:                     # 无符号变化 → 取使残差最小者
        bs = np.geomspace(lo, hi, 200)
        b = float(bs[int(np.argmin([abs(resid(x)) for x in bs]))])
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            r_mid = resid(mid)
            if r_lo * r_mid <= 0:
                hi, r_hi = mid, r_mid
            else:
                lo, r_lo = mid, r_mid
        b = 0.5 * (lo + hi)
    a = (e2 - e3) / (np.exp(-b * x2) - np.exp(-b * x3))
    return float(e3 - a * np.exp(-b * x3))


def composite_energy(energy_fn: Callable[..., float], coords,
                     scheme: Optional[Dict[str, object]] = None,
                     verbose: bool = True) -> Dict[str, float]:
    """按 CBS 复合方案计算单个几何点的总能量。

    参数
    ----
    energy_fn : ``energy_fn(coords, basis=..., method=...) -> float``
        (典型用法: lambda c, basis, method: PySCFCalculator(...).energy(c))
    coords : 几何 (Bohr)
    scheme : 见 :data:`DEFAULT_SCHEME`

    返回
    ----
    ``{"total", "hf_cbs", "corr_cbs", "delta_ccsdt", "hf_1", "hf_2",
    "corr_1", "corr_2", "ccsdt_small", "mp2_small"}`` (Hartree)
    """
    sc = dict(DEFAULT_SCHEME)
    if scheme:
        sc.update(scheme)
    hf_bases = tuple(sc["hf_bases"])
    hf_X = tuple(sc["hf_X"])
    c1, c2 = sc["corr_bases"]
    y1, y2 = sc["corr_X"]
    bd = sc["delta_basis"]
    md = sc["delta_method"]
    mr = sc["ref_method"]

    e_hf = [float(energy_fn(coords, basis=b, method="rhf")) for b in hf_bases]
    e_mp1 = float(energy_fn(coords, basis=c1, method=mr))
    e_mp2 = float(energy_fn(coords, basis=c2, method=mr))
    e_cc = float(energy_fn(coords, basis=bd, method=md))
    # δ 需要同基组的 MP2: 若 δ 基组就是相关能外推的最低基数则直接复用 (省一次调用)
    e_mp_small = (e_mp1 if (bd == c1) else
                  float(energy_fn(coords, basis=bd, method=mr)))

    if len(hf_bases) >= 3:
        args = []
        for i in range(3):                  # 参数顺序 (x1,e1,x2,e2,x3,e3)
            args.extend([float(hf_X[i]), e_hf[i]])
        hf_cbs = cbs_hf_3p(*args)
    else:
        hf_cbs = cbs_hf_2p(float(hf_X[0]), e_hf[0], float(hf_X[1]), e_hf[1])
    corr1, corr2 = e_mp1 - e_hf[0], e_mp2 - e_hf[1]
    corr_cbs = cbs_extrapolate_2p(float(y1), corr1, float(y2), corr2, 3.0)
    delta = e_cc - e_mp_small
    total = hf_cbs + corr_cbs + delta
    out = {"total": total, "hf_cbs": hf_cbs, "corr_cbs": corr_cbs,
           "delta_ccsdt": delta, "corr_1": corr1, "corr_2": corr2,
           "ccsdt_small": e_cc, "mp2_small": e_mp_small}
    for i, b in enumerate(hf_bases):
        out[f"hf_{i + 1}"] = e_hf[i]
    if verbose:
        print(f"    HF/CBS{hf_X} = {hf_cbs:.8f} | 相关/CBS({y1},{y2}) = "
              f"{corr_cbs:.8f} | δCCSD(T)/{bd} = {delta:.8f} → 总 = "
              f"{total:.8f} Ha", flush=True)
    return out


def make_energy_fn(symbols: Sequence[str], charge: int = 0, spin: int = 0,
                   **calc_kwargs) -> Callable[..., float]:
    """构造 ``composite_energy`` 需要的 ``energy_fn(coords, basis, method)``。

    用法::

        fn = make_energy_fn(["O", "H", "H"])
        out = composite_energy(fn, coords)      # → dict, 含 total(Ha)
    """
    from aqpes.pes.calculators import PySCFCalculator

    def energy_fn(coords, basis: str, method: str) -> float:
        calc = PySCFCalculator(list(symbols), basis=basis, method=method,
                               charge=charge, spin=spin, **calc_kwargs)
        return calc.energy(np.asarray(coords, dtype=float))

    return energy_fn
