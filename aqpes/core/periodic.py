"""元素数据: 同位素质量 (用于振动频率的约化质量) 与符号规范化。

质量单位: 原子质量单位 (amu)。数值取最常见同位素的 CODATA 质量
(与主流量子化学程序一致, 便于与文献频率逐位比对)。
未收录元素可传入自定义质量表 (``mass(sym, extra=...)``)。
"""

from __future__ import annotations

from typing import Dict, Optional

#: 最常见同位素质量 (amu) — 与 Gaussian/Molpro 默认使用的同位素一致
ISOTOPE_MASS: Dict[str, float] = {
    "H": 1.0078250319, "D": 2.0141017778, "He": 4.0026032497,
    "Li": 7.0160034366, "Be": 9.012183065, "B": 11.00930536,
    "C": 12.0, "N": 14.00307400443, "O": 15.99491461957,
    "F": 18.99840316273, "Ne": 19.9924401762, "Na": 22.9897692820,
    "Mg": 23.985041697, "Al": 26.98153853, "Si": 27.97692653465,
    "P": 30.97376199842, "S": 31.9720711744, "Cl": 34.968852682,
    "Ar": 39.9623831237, "K": 38.9637064864, "Ca": 39.962590863,
    "Sc": 44.95590828, "Ti": 47.94794198, "V": 50.94395704,
    "Cr": 51.94050623, "Mn": 54.93804391, "Fe": 55.93493633,
    "Co": 58.93319429, "Ni": 57.93534241, "Cu": 62.92959772,
    "Zn": 63.92914201, "Ga": 68.9255735, "Ge": 73.921177761,
    "As": 74.92159457, "Se": 79.9165218, "Br": 78.9183376,
    "Kr": 83.9114977282,
}

#: 原子质量单位 → 电子质量 (m_e), CODATA
AMU_TO_ME = 1822.888486209


def mass(symbol: str, extra: Optional[Dict[str, float]] = None) -> float:
    """返回元素最常见同位素质量 (amu)。

    Parameters
    ----------
    symbol : str
        元素符号 (大小写敏感, 如 ``"H"``, ``"Cl"``); ``"D"`` 为氘。
    extra : dict, optional
        额外/覆盖的质量表。
    """
    table = dict(ISOTOPE_MASS)
    if extra:
        table.update(extra)
    if symbol not in table:
        raise KeyError(
            f"未知元素 {symbol!r}; 已收录 {sorted(table)}; "
            f"可用 extra={{'{symbol}': <质量/amu>}} 传入")
    return table[symbol]


def masses(symbols, extra: Optional[Dict[str, float]] = None):
    """按符号序列返回质量数组 (amu)。"""
    return [mass(s, extra) for s in symbols]
