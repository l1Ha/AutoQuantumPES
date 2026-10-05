"""核心基础 (PES 子集): 量子体系描述、周期表常量、验证/工件工具。

注: 完整版 autoquantum 的 ``core/engine.py`` (端到端流水线, 依赖含时波包
动力学与可视化) 与 ``core/validation.py`` 中的传播诊断
(``PropagationHealth``/``check_propagation``/``suggest_dt``) **不在**本拆分
包内 —— 本包只覆盖势能面计算与拟合。
"""
from .base import QuantumSystem, PES
from .validation import (
    ValidationError, validate_grid, validate_energy_window,
    validate_pes_values, validate_positive, data_fingerprint,
    environment_info, write_run_manifest,
)

__all__ = [
    "QuantumSystem", "PES",
    "ValidationError", "validate_grid", "validate_energy_window",
    "validate_pes_values", "validate_positive", "data_fingerprint",
    "environment_info", "write_run_manifest",
]
