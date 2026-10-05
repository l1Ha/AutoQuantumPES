"""输入验证与数值健康诊断 — 科学计算软件的质量底线。

三类检查:
1. **输入验证** (validate_*): 网格/能量窗/势能值/参数的边界与有限性,
   失败时抛出带可操作提示的 ``ValidationError``;
2. **传播健康** (PropagationHealth): 波包传播后的守恒、吸收、能量漂移
   诊断 — 这些是量子动力学结果可信度的第一道闸门;
3. **步长建议** (suggest_dt): 基于势能幅度与 Nyquist 动能的启发式
   相位精度上限 (split-operator 无 CFL 限制, 但 dt 过大时相位误差
   与 CAP 吸收效率都会恶化)。

设计原则: 验证失败立即报错 (fail fast), 健康检查只告警不中断 —
物理上可疑但可继续的情形由调用方决定。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class ValidationError(ValueError):
    """输入验证失败 (含修复提示)。"""


# ---------------------------------------------------------------------------
# 输入验证
# ---------------------------------------------------------------------------

def validate_grid(name: str, grid: Sequence[float], min_points: int = 2,
                  min_length: float = 1e-12) -> np.ndarray:
    """验证等距有限网格: 返回 float64 数组。"""
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1:
        raise ValidationError(f"{name} 必须是 1D 数组, 实际 ndim={g.ndim}")
    if g.size < min_points:
        raise ValidationError(
            f"{name} 至少需要 {min_points} 个点, 实际 {g.size}")
    if not np.all(np.isfinite(g)):
        bad = int(np.sum(~np.isfinite(g)))
        raise ValidationError(f"{name} 含 {bad} 个非有限值 (NaN/Inf)")
    d = np.diff(g)
    if np.any(d <= 0):
        raise ValidationError(f"{name} 必须严格递增 (等距且 d>0)")
    spacing = float(np.max(np.abs(d - d[0])))
    if spacing > 1e-6 * max(abs(float(d[0])), 1e-30):
        raise ValidationError(
            f"{name} 非等距网格 (间距变化 {spacing:.2e}); "
            "FFT 传播要求严格等距")
    if abs(float(d[0])) < min_length:
        raise ValidationError(f"{name} 网格间距过小 ({d[0]:.2e})")
    return g


def validate_energy_window(e_min: float, e_max: float,
                           require_positive: bool = True) -> Tuple[float, float]:
    """验证能量扫描窗口: 升序、可选正能量。"""
    e_min, e_max = float(e_min), float(e_max)
    if not (np.isfinite(e_min) and np.isfinite(e_max)):
        raise ValidationError(f"能量窗含非有限值: [{e_min}, {e_max}]")
    if e_max <= e_min:
        raise ValidationError(
            f"能量窗必须升序: e_min={e_min} >= e_max={e_max}")
    if require_positive and e_min <= 0:
        raise ValidationError(
            f"散射/反应扫描要求正碰撞能, 收到 e_min={e_min}")
    return e_min, e_max


def validate_pes_values(name: str, V: np.ndarray,
                        max_dynamic_range: float = 1e8) -> np.ndarray:
    """验证势能面数值: 有限性 + 动态范围 (发散势能墙会毁掉 NN 拟合)。"""
    V = np.asarray(V, dtype=float)
    if not np.all(np.isfinite(V)):
        bad = int(np.sum(~np.isfinite(V)))
        raise ValidationError(
            f"{name} 含 {bad} 个非有限值; 检查势函数定义域 (如负键长/负距离)")
    span = float(V.max() - V.min())
    if span > max_dynamic_range:
        raise ValidationError(
            f"{name} 动态范围 {span:.3e} 超过 {max_dynamic_range:.0e}; "
            "势能墙发散 — 请截断网格或使用解析近核势")
    return V


def validate_positive(name: str, value: float, allow_zero: bool = False) -> float:
    """验证标量参数为有限的正数。

    Args:
        name: 参数名 (用于错误信息)。
        value: 待验证值。
        allow_zero: 是否允许 0。

    Returns:
        float: 验证通过后的值。

    Raises:
        ValidationError: 值非有限、为负, 或为 0 且不允许。
    """
    v = float(value)
    if not np.isfinite(v):
        raise ValidationError(f"{name} 非有限: {value}")
    if v < 0 or (v == 0 and not allow_zero):
        raise ValidationError(f"{name} 必须为正, 收到 {v}")
    return v


# ---------------------------------------------------------------------------
# 传播健康诊断
# ---------------------------------------------------------------------------

@dataclass
class PropagationHealth:
    """波包传播后的可信度诊断。"""

    norm_final: float
    absorbed_total: float
    channel_sum: Optional[float]
    energy_drift_rel: Optional[float] = None
    nan_detected: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """诊断是否通过 (无 NaN 且无任何告警)。"""
        return not self.nan_detected and not self.warnings

    def summary(self) -> str:
        """单行人可读摘要 (存活概率/吸收量/通道和/能量漂移)。"""
        parts = [f"norm={self.norm_final:.4f}", f"absorbed={self.absorbed_total:.4f}"]
        if self.channel_sum is not None:
            parts.append(f"P_react+P_refl={self.channel_sum:.6f}")
        if self.energy_drift_rel is not None:
            parts.append(f"dE/E={self.energy_drift_rel:.2e}")
        return ", ".join(parts)


def check_propagation(result, energy_drift_rel: Optional[float] = None,
                      residual_tol: float = 1e-6,
                      unabsorbed_tol: float = 0.05,
                      drift_tol: float = 1e-3) -> PropagationHealth:
    """诊断 WavePacket2DResult: NaN/守恒/未吸收比例/能量漂移。

    阈值语义:
    - ``residual_tol``: |存活 + Σ吸收 − 1| 容差 (记账恒等式);
    - ``unabsorbed_tol``: 末态网格存活概率上限 — 过大说明传播时长不足
      或 CAP 太弱, 反应概率会系统性偏低;
    - ``drift_tol``: ⟨H⟩ 相对漂移上限 (仅在提供时检查)。
    """
    norm = np.asarray(result.norm_t, dtype=float)
    nan = (not np.all(np.isfinite(norm))
           or not np.all(np.isfinite(result.reaction_prob)))
    absorbed = float(sum(np.asarray(v)[-1]
                         for v in result.absorbed.values())) if result.absorbed else 0.0
    warnings: List[str] = []

    identity = float(norm[-1] + absorbed)
    if not nan and abs(identity - 1.0) > residual_tol:
        warnings.append(f"概率记账偏差 {identity - 1.0:+.2e} 超出容差 {residual_tol:.0e}")

    channel_sum = None
    if result.reaction_prob is not None and result.reflection_prob is not None:
        channel_sum = float(result.reaction_prob[-1] + result.reflection_prob[-1])
        if not nan and abs(channel_sum - 1.0) > residual_tol:
            warnings.append(
                f"通道和偏离 1 ({channel_sum:.6f}); 检查掩码是否互补划分网格")

    if not nan and norm[-1] > unabsorbed_tol:
        warnings.append(
            f"末态仍有 {norm[-1]:.1%} 概率留在网格 (阈值 {unabsorbed_tol:.0%}): "
            "传播时间不足或 CAP 太弱, 反应概率可能偏低 — 增大 wp_n_steps "
            "或 cap_height")

    if energy_drift_rel is not None and not nan:
        if abs(energy_drift_rel) > drift_tol:
            warnings.append(
                f"能量相对漂移 {energy_drift_rel:+.2e} 超过 {drift_tol:.0e}: "
                "减小 dt 或检查势能突变")

    if nan:
        warnings.append("波函数出现 NaN/Inf: 检查 dt、势能幅度与初始波包")

    return PropagationHealth(
        norm_final=float(norm[-1]), absorbed_total=absorbed,
        channel_sum=channel_sum, energy_drift_rel=energy_drift_rel,
        nan_detected=bool(nan), warnings=warnings)


# ---------------------------------------------------------------------------
# 步长与网格建议
# ---------------------------------------------------------------------------

def suggest_dt(V: np.ndarray, dR: float, dr: float,
               mass_R: float, mass_r: float,
               c: float = 0.5) -> float:
    """启发式最大步长 (au): 兼顾势能相位与 Nyquist 动能相位精度。

    split-operator 传播无 CFL 限制, 但每步相位增量 Δφ = dt·E 应远
    小于 1: E_max ≈ max(|V|_max, E_kin(Nyquist))。取 dt ≤ c/E_max。
    """
    V = np.asarray(V, dtype=float)
    v_amp = float(np.max(np.abs(V)))
    kR = np.pi / abs(float(dR))
    kr = np.pi / abs(float(dr))
    e_kin = 0.5 * max(kR ** 2 / mass_R, kr ** 2 / mass_r)
    e_max = max(v_amp, e_kin)
    if e_max <= 0:
        return float("inf")
    return float(c / e_max)


# ---------------------------------------------------------------------------
# 可复现性: 运行清单与数据指纹
# ---------------------------------------------------------------------------

def data_fingerprint(*arrays: np.ndarray) -> str:
    """数组内容的 SHA-256 指纹 (形状+dtype+字节), 用于数据溯源。"""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def environment_info() -> Dict[str, Any]:
    """运行环境快照 (版本/平台), 写入运行清单。"""
    import numpy
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
    }
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    try:
        from aqpes import __version__ as aq_version
        info["aqpes"] = aq_version
    except Exception:
        pass
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        if commit:
            info["git_commit"] = commit
    except Exception:
        pass
    return info


def _config_to_dict(config: Any) -> Dict[str, Any]:
    """尽力提取配置为普通 dict。

    注: Python 3.14 中函数内定义的类其 __dict__ 代理可能为 falsy 且
    isinstance(C, type) 不可靠, 因此不依赖真值判断, 逐级回退。
    """
    try:
        d = {k: v for k, v in vars(config).items() if not k.startswith("__")}
        if len(d) > 0:
            return d
    except TypeError:
        pass
    try:
        d = {k: getattr(config, k) for k in dir(config)
             if not k.startswith("_") and not callable(getattr(config, k, None))}
        if len(d) > 0:
            return d
    except Exception:
        pass
    return {"config": str(config)}


def write_run_manifest(path: str, config: Any, results_summary: Dict[str, Any],
                       extra: Optional[Dict[str, Any]] = None) -> str:
    """写运行清单 (配置+环境+结果摘要+数据指纹) — 结果可复现的凭证。"""
    cfg = _config_to_dict(config)
    manifest = {
        "environment": environment_info(),
        "config": cfg,
        "results": results_summary,
    }
    if extra:
        manifest.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    return path
