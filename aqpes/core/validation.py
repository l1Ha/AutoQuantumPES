"""输入验证与可复现性工件 — 科学计算软件的质量底线。

两类检查:
1. **输入验证** (``validate_*``): 网格/能量窗/势能值/参数的边界与有限性,
   失败时抛出带可操作提示的 ``ValidationError``;
2. **可复现性** (``data_fingerprint`` / ``environment_info`` /
   ``write_run_manifest``): 数据集指纹与运行环境清单, 保证势能面数据与
   拟合结果可溯源。

设计原则: 验证失败立即报错 (fail fast)。
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
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
        info["autoquantum"] = aq_version
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
