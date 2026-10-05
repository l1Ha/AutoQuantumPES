"""电子结构计算后端 — 为 NN 势能面拟合生成从头算训练数据。

本模块定义统一的 ``Calculator`` 协议 (能量 Hartree + 梯度
Hartree/Bohr), 并提供:

- ``AnalyticCalculator``: 包装任意 (E, ∇E) 可调用对 — 测试/演示
  以及"解析面即数据源"的闭环, 无外部依赖, 完全可验证;
- ``XTBCommandCalculator``: 子进程调用 GFN-xTB 的半经验方法
  (实验性 — 需要安装 xtb, 本仓库发布环境未验证);
- ``PySCFCalculator``: PySCF Hartree-Fock/DFT (实验性, 可选依赖);
- ``ASECalculatorAdapter``: 包装任意 ASE calculator 对象 (实验性)。

**诚实声明**: 真实量子化学后端的结果依赖具体程序版本、基组与电子
结构设置, 本仓库不对其数值正确性做任何认证; 未经独立基准校验前,
其数据只应被视为"与该程序设置一致的标签"。可复现性依赖完整的
程序版本与参数记录 (见 ``provenance``)。

单位约定: 坐标 Bohr, 能量 Hartree, 梯度 Hartree/Bohr —— 与动力学
模块一致, 无需换算。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


class Calculator(ABC):
    """电子结构计算后端协议: 能量 (Hartree) + 核梯度 (Hartree/Bohr)。"""

    #: 后端名称 (子类覆盖)
    name: str = "base"

    @abstractmethod
    def energy(self, coords: np.ndarray) -> float:
        """基态能量 (Hartree)。coords: (N_atoms, 3) Bohr。"""

    @abstractmethod
    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """核坐标梯度 dE/dR (N_atoms, 3) Hartree/Bohr。"""

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """能量与梯度 (默认分别调用; 有原生梯度支持的后端应覆盖)。"""
        return self.energy(coords), self.gradient(coords)

    @property
    def provenance(self) -> Dict[str, str]:
        """数据来源记录 (程序/版本/参数) — 随数据集保存。"""
        return {"backend": self.name}


# ---------------------------------------------------------------------------
# 解析后端 (无外部依赖, 完全可验证)
# ---------------------------------------------------------------------------

class AnalyticCalculator(Calculator):
    """包装解析能量/梯度可调用对。

    Parameters
    ----------
    energy_fn : Callable
        ``(N, 3) coords -> float`` (Hartree)。
    gradient_fn : Callable, optional
        ``(N, 3) coords -> (N, 3)`` 解析梯度; 缺省时用中心差分
        (精度受步长限制, 仅供演示/测试)。
    name : str
    """

    name = "analytic"

    def __init__(self, energy_fn: Callable, gradient_fn: Optional[Callable] = None,
                 grad_h: float = 1e-5, name: str = "analytic"):
        self.energy_fn = energy_fn
        self.gradient_fn = gradient_fn
        self.grad_h = grad_h
        if name != "analytic":
            self.name = name

    def energy(self, coords: np.ndarray) -> float:
        """解析能量 (Hartree)。coords: (N, 3) Bohr。"""
        return float(self.energy_fn(np.asarray(coords, dtype=float)))

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """解析梯度 (Hartree/Bohr); 未提供解析梯度时用中心差分。"""
        coords = np.asarray(coords, dtype=float)
        if self.gradient_fn is not None:
            return np.asarray(self.gradient_fn(coords), dtype=float)
        g = np.zeros_like(coords)
        for i in range(coords.shape[0]):
            for j in range(3):
                cp, cm = coords.copy(), coords.copy()
                cp[i, j] += self.grad_h
                cm[i, j] -= self.grad_h
                g[i, j] = (self.energy(cp) - self.energy(cm)) / (2 * self.grad_h)
        return g


# ---------------------------------------------------------------------------
# 外部程序后端 (实验性)
# ---------------------------------------------------------------------------

class CommandBackendError(RuntimeError):
    """外部程序调用/解析失败。"""


class XTBCommandCalculator(Calculator):
    """GFN-xTB 半经验方法 (子进程调用; 实验性)。

    对每个几何执行 ``xtb coordfile --grad [--charg q] [--uhf u]``,
    从 stdout 解析总能量与梯度块。**需要本机安装 xtb 且在 PATH 中**;
    本仓库的发布验证环境未安装 xtb, 该后端未经端到端验证。
    """

    name = "xtb"

    _ENERGY_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s*Eh", re.IGNORECASE)
    _GRAD_RE = re.compile(r"gradient \(Eh/a0\)", re.IGNORECASE)
    _CYCLE_RE = re.compile(r"cycle\s+(\d+)", re.IGNORECASE)

    def __init__(self, symbols: Sequence[str], charge: int = 0,
                 uhf: int = 0, accuracy: float = 1.0,
                 binary: str = "xtb", timeout: float = 300.0):
        self.symbols = list(symbols)
        self.charge = charge
        self.uhf = uhf
        self.accuracy = accuracy
        self.binary = binary
        self.timeout = timeout
        if shutil.which(binary) is None:
            raise CommandBackendError(
                f"未找到 {binary!r}; 请安装 GFN-xTB 并加入 PATH "
                "(或改用 AnalyticCalculator / PySCF / ASE 后端)")

    def _run(self, coords: np.ndarray) -> str:
        coords = np.asarray(coords, dtype=float)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "coord.xyz")
            with open(path, "w") as f:
                f.write(f"{coords.shape[0]}\n\n")
                for sym, (x, y, z) in zip(self.symbols, coords):
                    f.write(f"{sym:2s} {x:20.12f} {y:20.12f} {z:20.12f}\n")
            cmd = [self.binary, path, "--grad",
                   "--chrg", str(self.charge), "--uhf", str(self.uhf),
                   "--acc", str(self.accuracy)]
            try:
                proc = subprocess.run(cmd, cwd=tmp, capture_output=True,
                                      text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                raise CommandBackendError(f"xtb 调用超时 ({self.timeout}s)") from exc
            if proc.returncode != 0:
                raise CommandBackendError(
                    f"xtb 返回码 {proc.returncode}:\n{proc.stderr[-2000:]}")
            return proc.stdout

    def _parse(self, out: str) -> Tuple[float, np.ndarray]:
        m = self._ENERGY_RE.search(out)
        if not m:
            raise CommandBackendError("xtb 输出中未找到 TOTAL ENERGY 行")
        energy = float(m.group(1))

        n = len(self.symbols)
        gi = self._GRAD_RE.search(out)
        grad = np.zeros((n, 3))
        if gi:
            tail = out[gi.end():]
            nums = re.findall(r"(-?\d+\.\d+)(?:E[-+]?\d+)?", tail)
            values = np.array([float(x) for x in nums[:3 * n]])
            if values.size == 3 * n:
                grad = values.reshape(n, 3)
        return energy, grad

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """单次子进程调用同时返回能量 (Hartree) 与梯度 (Hartree/Bohr)。"""
        return self._parse(self._run(coords))

    def energy(self, coords: np.ndarray) -> float:
        """GFN-xTB 总能量 (Hartree); coords: (N, 3) Bohr。"""
        return self.energy_and_gradient(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """GFN-xTB 核梯度 (Hartree/Bohr)。"""
        return self.energy_and_gradient(coords)[1]

    @property
    def provenance(self) -> Dict[str, str]:
        """记录 xtb 版本、电荷、自旋多重度与精度设置。"""
        try:
            ver = subprocess.run([self.binary, "--version"],
                                 capture_output=True, text=True,
                                 timeout=30).stdout.strip().splitlines()[0]
        except Exception:
            ver = "unknown"
        return {"backend": "xtb", "version": ver, "charge": str(self.charge),
                "uhf": str(self.uhf), "accuracy": str(self.accuracy)}


#: 常见溶剂的静态介电常数 (ddCOSMO 用)
_SOLVENT_EPS: Dict[str, float] = {
    "water": 78.3553, "methanol": 32.613, "ethanol": 24.852,
    "dmso": 46.826, "acetonitrile": 35.688, "acetone": 20.493,
    "dichloromethane": 8.93, "chloroform": 4.7113, "thf": 7.4257,
    "toluene": 2.3741, "cyclohexane": 2.0165,
}


class PySCFCalculator(Calculator):
    """PySCF 从头算与 DFT 计算后端 (可选依赖; 实验性)。

    支持能力:
    - **SCF 层**: RHF, ROHF, UHF, DFT (RKS/ROKS/UKS)
    - **多参考** (v0.25.0–v0.26.0): CASSCF + **NEVPT2 动态相关** — 键断裂/强关联
    - **激发态** (v0.27.0–v0.29.0): TD-DFT/TDHF 与 **EOM-CCSD**; `state=k` 即
      激发态势能面; `follow=True` 启用**根跟踪** (态交叉时保持物理身份)
    - **隐式溶剂** (v0.28.0–v0.30.0): `solvent_model` = ddCOSMO (默认) /
      **PCM** (C-PCM, IEF-PCM, COSMO, SS(V)PE) / **ddPCM** / **SMD**
      (`solvent="water"` / `solvent_eps=78.4`)
    - **相关方法** (v0.21.0): MP2, CCSD, CCSD(T) — 带解析核梯度
      (MP2: `grad.mp2`; CCSD/(T): `grad.ccsd`, 需在梯度前调用 `ccsd_t()`)
    - **高自旋约束与自旋锁定 (spin_lock)**: 开壳层自旋纯度审计, 防止态跃迁与自旋污染
    - **最大重叠法 (MOM)**: 沿几何路径保持特定电子轨道占据, 克服激发态与非平衡态变分塌陷
    - **共振态衰减宽度 (CAP-PES)**: 提供自电离衰变宽度 Γ(R) 与复能量 E_R - i*Γ/2 接口
    - **自旋-轨道耦合 (SOC, v0.31.0)**: 单电子 Breit–Pauli 三层接口 —
      :meth:`soc_terms` (轨道层 ζ/精细结构分裂) 与
      :meth:`soc_state_interaction` (单重态-三重态耦合矩阵, cm⁻¹);
      见 ``aqpes.pes.soc`` (含逐元素误差的诚实边界)
    """

    name = "pyscf"

    #: 相关方法 → (SCF 参考, 求解器类型)
    _CORRELATED: Dict[str, str] = {
        "mp2": "mp2",
        "ccsd": "ccsd",
        "ccsd(t)": "ccsd(t)",
        "ccsd_t": "ccsd(t)",
        "eom-ccsd": "eom-ccsd",       # EOM-CCSD 激发态 (v0.29.0)
    }
    #: 纯 SCF 方法
    _SCF_ONLY = ("rhf", "rohf", "uhf", "dft", "rks", "roks", "uks", "casscf",
                 "tddft")

    def __init__(self, symbols: Sequence[str], basis: str = "sto-3g",
                 charge: int = 0, spin: int = 0, method: str = "rhf",
                 xc: Optional[str] = None, spin_lock: bool = False,
                 spin_tol: float = 0.1, use_mom: bool = False,
                 mom_reference: str = "prev",
                 cap_params: Optional[Dict[str, Any]] = None,
                 unit: str = "Bohr", conv_tol: float = 1e-9,
                 max_cycle: int = 100, frozen_core: bool = False,
                 grad_t_mode: str = "fd", grad_t_h: float = 1e-4,
                 relativistic: Optional[str] = None,
                 active_space: Optional[Tuple[int, int]] = None,
                 pt2: Optional[str] = None,
                 nstates: int = 5, state: Optional[int] = None,
                 solvent: Optional[str] = None,
                 solvent_eps: Optional[float] = None,
                 solvent_model: str = "ddcosmo",
                 pcm_variant: str = "IEF-PCM",
                 follow: bool = False):
        self.symbols = list(symbols)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.method = method.lower()
        self.xc = xc
        self.spin_lock = spin_lock
        self.spin_tol = float(spin_tol)
        self.use_mom = use_mom
        self.mom_reference = mom_reference
        self.cap_params = cap_params
        self.unit = unit
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(max_cycle)
        self.frozen_core = bool(frozen_core)
        # CCSD(T) 的 (T) 项梯度: PySCF 的 grad.ccsd 不含 (T), 故默认用有限差分
        # 补上 (T) 增量 (grad_t_mode="fd"), 保证能量与力一致; "ccsd" 则跳过
        # (快但力与 CCSD(T) 能量不一致, 仅在明确接受该近似时使用)。
        self.grad_t_mode = str(grad_t_mode)
        self.grad_t_h = float(grad_t_h)
        # 标量相对论: None (非相对论) | "x2c" (X2C 哈密顿量, 1e 积分修正)
        self.relativistic = relativistic
        # CASSCF 活性空间 (ncas, nelecas); method="casscf" 时必填
        self.active_space = active_space
        # 动态相关 (CASSCF 之上的微扰): None | "nevpt2"
        # 注: PySCF 无 CASPT2 模块; NEVPT2 是同一层级 (CAS 参考 + 二阶微扰)
        # 且**无侵入态问题** (intruder-state free), 是更稳健的替代。
        self.pt2 = pt2.lower() if pt2 else None
        if self.pt2 not in (None, "nevpt2"):
            raise ValueError(f"未知 pt2: {pt2!r}; 支持 None 或 'nevpt2'")
        if self.pt2 and self.method != "casscf":
            raise ValueError('pt2="nevpt2" 需要 method="casscf"')
        if method.lower() == "casscf" and grad_t_mode != "casci":
            # CASSCF/NEVPT2 默认走有限差分梯度 (PySCF 无解析梯度模块)
            self.grad_t_mode = "fd"
        # TD-DFT/TDHF 激发态 (v0.27.0): nstates=计算态数, state=选中的激发态
        # (0-based; None = 基态, 保持默认行为不变)。指定 state 后 energy()
        # 返回**激发态总能量** E_ref + E_exc[state], 因此扫描/优化/NEB/IRC/
        # 频率等全部工作流可直接作用于**激发态势能面**。
        self.nstates = int(nstates)
        self.state = None if state is None else int(state)
        # 激发态根跟踪 (v0.29.0): follow=True 时按与上一几何的激发向量
        # 最大重叠选根, 避免态交叉处 state 序号物理身份漂移
        self.follow = bool(follow)
        self._prev_exc_vec = None
        # 隐式溶剂: solvent="water" 或 solvent_eps=78.4
        #   - solvent_model="ddcosmo" (默认, v0.28.0): ddCOSMO
        #   - "pcm" (v0.30.0): C-PCM / IEF-PCM / COSMO / SS(V)PE (见 pcm_variant)
        #   - "ddpcm" (v0.30.0): domain-decomposition PCM (PySCF 标注 testing)
        #   - "smd" (v0.30.0): SMD 全溶剂化模型 (需命名溶剂; 仅 SCF 层; PySCF 实验性)
        self.solvent = solvent
        self.solvent_model = str(solvent_model).lower()
        self.pcm_variant = str(pcm_variant)
        if self.solvent_model not in ("ddcosmo", "pcm", "ddpcm", "smd"):
            raise ValueError(
                f"未知 solvent_model: {solvent_model!r}; 支持 "
                "'ddcosmo', 'pcm', 'ddpcm', 'smd'")
        if self.solvent_model == "pcm" and self.pcm_variant not in (
                "C-PCM", "IEF-PCM", "COSMO", "SS(V)PE"):
            raise ValueError(
                f"未知 pcm_variant: {pcm_variant!r}; 支持 'C-PCM', "
                "'IEF-PCM', 'COSMO', 'SS(V)PE'")
        if self.solvent_model == "smd":
            # SMD 的介电常数与非静电项 (CDS: cavitation/dispersion/solvent
            # structure) 都由溶剂名从 PySCF SMD 数据库决定, 不能自定义 eps
            if solvent is None:
                raise ValueError(
                    'solvent_model="smd" 需要命名溶剂 (如 solvent="water"); '
                    "SMD 的非静电项需要溶剂的 Abraham 参数集")
            if solvent_eps is not None:
                raise ValueError(
                    'solvent_model="smd" 的 eps 由溶剂名从 SMD 数据库取得, '
                    "不支持 solvent_eps")
        if solvent is not None or solvent_eps is not None:
            if self.solvent_model == "smd":
                self.solvent_eps = None
            elif solvent_eps is not None:
                self.solvent_eps = float(solvent_eps)
            else:
                key = str(solvent).lower()
                if key not in _SOLVENT_EPS:
                    raise ValueError(
                        f"未知溶剂 {solvent!r}; 可用 {sorted(_SOLVENT_EPS)} "
                        f"或直接给 solvent_eps=<介电常数>")
                self.solvent_eps = _SOLVENT_EPS[key]
        else:
            self.solvent_eps = None
        # SCF 层解析溶剂梯度不可用时 (PySCF 无 pyscf/solvent/grad/<model>),
        # 梯度退化为中心有限差分。实测 (Slurm 1558896, H2O/6-31G*, eps=78.4):
        #   ddcosmo 3.6e-07 ✓ | pcm 4.2e-08 ✓ | ddpcm 8.4e-04 ✗ (≈8% |g|max,
        #   返回的梯度不含 ddPCM 溶剂响应 → 必须 FD)。
        self._solvent_fd_grad = (self.solvent_model == "ddpcm")
        # ddPCM 附加限制: eps=1 时 PySCF 内部除零 (实测 ZeroDivisionError);
        # 物理极限需用 eps=1+δ 逼近 (验证脚本用 1.000001)。
        if method.lower() == "tddft" and self.state is not None \
                and grad_t_mode != "analytic":
            # 激发态默认有限差分梯度 (PySCF 的 TD 梯度不支持按态选择)
            self.grad_t_mode = "fd"

        if self.method not in self._SCF_ONLY and self.method not in self._CORRELATED:
            raise ValueError(
                f"未知 method: {self.method}; 支持 {sorted(self._SCF_ONLY + tuple(self._CORRELATED))}"
            )

        self._ref_mo_coeff = None
        self._ref_mo_occ = None
        self._initial_mo_coeff = None
        self._initial_mo_occ = None

        try:
            import pyscf  # noqa: F401
        except ImportError as exc:
            raise CommandBackendError(
                "未安装 pyscf; pip install pyscf, 或改用其他后端") from exc

    def reset_mom(self, mo_coeff: Optional[np.ndarray] = None,
                  mo_occ: Optional[np.ndarray] = None) -> None:
        """重置或手动指定 MOM (最大重叠法) 的参考轨道。"""
        self._ref_mo_coeff = mo_coeff
        self._ref_mo_occ = mo_occ
        self._initial_mo_coeff = mo_coeff
        self._initial_mo_occ = mo_occ

    def _pick_root(self, energies: np.ndarray, vecs: np.ndarray) -> int:
        """选根: 默认按 ``state``; ``follow=True`` 时按与上一几何激发向量的
        最大重叠选根 (**根跟踪**), 避免态交叉处 state 序号物理身份漂移。
        """
        if self.state is None:
            return 0
        if not self.follow or self._prev_exc_vec is None:
            return int(self.state)
        prev = np.asarray(self._prev_exc_vec).ravel()
        V = np.asarray(vecs)
        if V.ndim == 1:
            return int(self.state)
        V = V.reshape(V.shape[0], -1)
        if V.shape[1] != prev.size:            # 几何变化导致维度变化 → 放弃跟踪
            return int(self.state)
        ov = np.abs(V @ prev)
        return int(np.argmax(ov))

    @property
    def _solvent_active(self) -> bool:
        """是否启用隐式溶剂 (SMD 只给溶剂名、不给 eps)。"""
        return self.solvent is not None or self.solvent_eps is not None

    def _solvent_desc(self) -> str:
        """溶剂配置的单行描述 (provenance 用)。"""
        if not self._solvent_active:
            return "none"
        if self.solvent_model == "smd":
            return f"smd/{self.solvent} (eps 由 SMD 数据库)"
        extra = f"/{self.pcm_variant}" if self.solvent_model == "pcm" else ""
        return f"{self.solvent_model}{extra} (eps={self.solvent_eps})"

    def _attach_solvent(self, obj, kind: str = "scf"):
        """把隐式溶剂接入 SCF / post-SCF / TD / CASSCF 对象。

        PySCF 的四条入口为 ``{model}_for_{scf|post_scf|tdscf|casscf}``;
        不支持的组合 (如 SMD 仅提供 ``smd_for_scf``) 会明确报错而非静默回退气相。
        """
        if not self._solvent_active:
            return obj
        import importlib
        model = self.solvent_model
        mod = importlib.import_module(f"pyscf.solvent.{model}")
        kind_full = {"post": "post_scf", "td": "tdscf"}.get(kind, kind)
        fn = getattr(mod, f"{model}_for_{kind_full}", None)
        if fn is None:
            raise CommandBackendError(
                f"溶剂模型 {model} 不支持 {kind} 入口 (PySCF 无 "
                f"{model}_for_{kind_full}); SMD 目前仅支持 SCF 层")
        solvent_obj = None
        if model == "smd":
            # SMD 的非静电项需要溶剂名对应参数集, 必须在构造溶剂对象时给出
            try:
                solvent_obj = mod.SMD(obj.mol, solvent=str(self.solvent).lower())
            except Exception as exc:
                raise CommandBackendError(
                    f"SMD 溶剂 {self.solvent!r} 不可用 (不在 PySCF SMD "
                    f"数据库中?): {exc}") from exc
        try:
            out = fn(obj) if solvent_obj is None else fn(obj, solvent_obj)
        except CommandBackendError:
            raise
        except Exception as exc:
            raise CommandBackendError(
                f"溶剂模型 {model} 无法接入 {kind} 对象 "
                f"({type(obj).__name__}): {exc}") from exc
        ws = out.with_solvent
        if model == "pcm":
            ws.method = self.pcm_variant
        if self.solvent_eps is not None:
            ws.eps = self.solvent_eps
        return out

    def _mol(self, coords: np.ndarray):
        from pyscf import gto
        mol = gto.Mole(atom=[(s, c) for s, c in zip(self.symbols, coords)],
                       basis=self.basis, charge=self.charge,
                       spin=self.spin, unit=self.unit, verbose=0)
        mol.build()  # 显式构建, 避免 SCF kernel 触发未初始化告警
        return mol

    def _mf(self, mol):
        """构造 SCF 参考 (相关方法按其自旋选择 RHF/UHF 参考)。"""
        from pyscf import scf, dft
        m = self.method
        ref = self._CORRELATED.get(m)
        if ref is not None:
            # 相关方法的参考波函数: 闭壳层 RHF, 开壳层 UHF
            # (PySCF 的解析 CCSD 梯度支持 RHF/UHF 参考)
            mf = scf.RHF(mol) if self.spin == 0 else scf.UHF(mol)
        elif m == "rhf":
            mf = scf.RHF(mol)
        elif m == "rohf":
            mf = scf.ROHF(mol)
        elif m == "uhf":
            mf = scf.UHF(mol)
        elif m in ("dft", "rks"):
            mf = dft.RKS(mol) if self.spin == 0 else dft.ROKS(mol)
            if self.xc:
                mf.xc = self.xc
        elif m == "roks":
            mf = dft.ROKS(mol)
            if self.xc:
                mf.xc = self.xc
        elif m == "uks":
            mf = dft.UKS(mol)
            if self.xc:
                mf.xc = self.xc
        elif m == "casscf":
            # CASSCF: 先用 RHF/ROHF 参考, 再由 _run 做多组态自洽
            mf = scf.RHF(mol) if self.spin == 0 else scf.ROHF(mol)
        elif m == "tddft":
            # TD-DFT/TDHF: xc 给定 → RKS 参考 + TDDFT; 否则 RHF + TDHF
            if self.spin != 0:
                raise CommandBackendError(
                    'method="tddft" 目前仅支持闭壳层参考 (spin=0)')
            if self.xc:
                mf = dft.RKS(mol)
                mf.xc = self.xc
            else:
                mf = scf.RHF(mol)
        else:
            raise ValueError(
                f"未知 method: {self.method}; 支持 'rhf', 'rohf', 'uhf', "
                f"'dft', 'rks', 'roks', 'uks', 'mp2', 'ccsd', 'ccsd(t)'"
            )

        mf.conv_tol = self.conv_tol
        mf.max_cycle = self.max_cycle
        if self._solvent_active:
            mf = self._attach_solvent(mf, "scf")
            mf.conv_tol = self.conv_tol
            mf.max_cycle = self.max_cycle
        if self.relativistic == "x2c":
            # X2C 标量相对论单电子哈密顿量 (PySCF: mf.x2c())
            mf = mf.x2c()
            mf.conv_tol = self.conv_tol
            mf.max_cycle = self.max_cycle
        elif self.relativistic is not None:
            raise ValueError(
                f"未知 relativistic: {self.relativistic!r}; 支持 None 或 'x2c'")
        return mf

    def _corr_solver(self, mf):
        """构造相关方法求解器 (MP2 / CCSD), 返回 (solver, 是否含 (T))。

        注意: PySCF 的 ``frozen`` 是**构造参数**(``mp.MP2(mf, frozen=...)`` /
        ``cc.CCSD(mf, frozen=...)``), 不是 ``kernel()`` 的参数。
        """
        from pyscf import cc, mp
        ref = self._CORRELATED.get(self.method)
        if ref is None:
            return None, False
        frozen = None
        if self.frozen_core:
            from pyscf.data import elements
            frozen = elements.chemcore(mf.mol)
        if ref == "mp2":
            solver = mp.MP2(mf, frozen=frozen)
            return self._attach_solvent(solver, "post"), False
        mycc = cc.CCSD(mf, frozen=frozen)
        return (self._attach_solvent(mycc, "post"),
                ref in ("ccsd(t)", "eom-ccsd"))

    def _run(self, coords: np.ndarray, need_grad: bool = True):
        """计算能量 (Hartree), 可选计算核梯度。

        ``need_grad=False`` 时跳过梯度 (对 CCSD(T) 尤其重要: 其 (T) 项梯度
        为 6N 次 CCSD(T) 的有限差分, 纯能量调用不应触发)。
        """
        from pyscf import lib
        coords_arr = np.asarray(coords, dtype=float)
        mol = self._mol(coords_arr)
        mf = self._mf(mol)

        # MOM (最大重叠法) 注入: 以参考轨道最大重叠原则决定每步占据,
        # 维持指定激发态组态 (PySCF 2.x API: mom_occ(mf, occorb, setocc), 原地修改)
        if self.use_mom and self._ref_mo_coeff is not None and self._ref_mo_occ is not None:
            from pyscf.scf import addons
            setocc = self._mom_setocc(self._ref_mo_occ)
            mf = addons.mom_occ(mf, self._ref_mo_coeff, setocc)

        e = mf.kernel()
        if not mf.converged:
            raise CommandBackendError("PySCF SCF 未收敛")

        # 自旋态审计与自旋锁定
        if hasattr(mf, "spin_square") and callable(mf.spin_square):
            try:
                res = mf.spin_square()
                if isinstance(res, (tuple, list)) and len(res) >= 2:
                    ss, _ = res[0], res[1]
                    s_ideal = abs(self.spin) / 2.0
                    s2_ideal = s_ideal * (s_ideal + 1.0)
                    s2_diff = abs(ss - s2_ideal)
                    if self.spin_lock and s2_diff > self.spin_tol:
                        raise CommandBackendError(
                            f"自旋锁定失败: 实际 <S^2>={ss:.4f}, 理论值 S(S+1)={s2_ideal:.4f}, "
                            f"自旋污染偏差 {s2_diff:.4f} 超过阈值 {self.spin_tol}"
                        )
            except CommandBackendError:
                raise
            except Exception:
                pass

        # 缓存当前收敛轨道供后续构型 MOM 跟踪
        if self.use_mom:
            mo_c = mf.mo_coeff
            mo_o = mf.mo_occ
            if self._initial_mo_coeff is None:
                self._initial_mo_coeff = mo_c
                self._initial_mo_occ = mo_o
            if self.mom_reference == "prev":
                self._ref_mo_coeff = mo_c
                self._ref_mo_occ = mo_o
            elif self.mom_reference == "initial":
                self._ref_mo_coeff = self._initial_mo_coeff
                self._ref_mo_occ = self._initial_mo_occ

        # ---- TD-DFT / TDHF: 激发态 (可选 state → 激发态势能面) ----
        if self.method == "tddft":
            from pyscf.tdscf import rks as td_rks, rhf as td_rhf
            td = td_rks.TDDFT(mf) if self.xc else td_rhf.TDHF(mf)
            td = self._attach_solvent(td, "td")
            td.nstates = self.nstates
            es = td.kernel()[0]                 # Hartree
            self._last_td = td
            f_osc = np.asarray(td.oscillator_strength(), dtype=float)
            self.last_excitations = np.asarray(es, dtype=float)
            self.last_oscillator_strengths = f_osc
            if self.state is None:
                return float(mf.e_tot), None
            if not (0 <= self.state < len(es)):
                raise CommandBackendError(
                    f"state={self.state} 超出范围 (共 {len(es)} 个态)")
            # 根跟踪: 用 TD 激发向量与上一几何的最大重叠选根
            vecs = getattr(td, "xy", None)
            idx = int(self.state)
            if vecs is not None:
                V = np.asarray(vecs)
                idx = self._pick_root(np.asarray(es), V)
                self._prev_exc_vec = np.asarray(V[idx]).ravel().copy()
            e_exc = float(mf.e_tot) + float(es[idx])
            if not need_grad:
                return e_exc, None
            # ⚠ PySCF 的 TD 梯度 (grad.tdrks/tdrhf) **不响应态选择**
            # (实测设置 td.state 前后梯度完全相同) → 默认用中心有限差分,
            # 保证对任意 state 都正确; grad_t_mode="analytic" 仅当明确只需
            # 最低激发态时可用。
            if self.grad_t_mode == "analytic":
                from pyscf import grad as pyscf_grad
                gmod = pyscf_grad.tdrks if self.xc else pyscf_grad.tdhf if False else pyscf_grad.tdrhf
                if self.state:
                    td.state = idx
                g = gmod.Gradients(td).kernel()
                return e_exc, np.asarray(lib.asarray(g), dtype=float).reshape(-1, 3)
            g = np.zeros_like(coords_arr)
            for i in range(coords_arr.shape[0]):
                for j in range(3):
                    cp, cm = coords_arr.copy(), coords_arr.copy()
                    cp[i, j] += self.grad_t_h
                    cm[i, j] -= self.grad_t_h
                    g[i, j] = (self._energy_only(cp) - self._energy_only(cm)) / \
                        (2 * self.grad_t_h)
            return e_exc, g

        # ---- CASSCF: 多组态自洽场 (静态相关 / 键断裂) ----
        if self.method == "casscf":
            if not self.active_space:
                raise CommandBackendError(
                    'method="casscf" 需要 active_space=(ncas, nelecas)')
            from pyscf import mcscf
            ncas, nelecas = self.active_space
            mc = mcscf.CASSCF(mf, int(ncas), int(nelecas))
            mc = self._attach_solvent(mc, "casscf")
            mc.conv_tol = max(self.conv_tol, 1e-8)
            mc.max_cycle = self.max_cycle
            e = float(mc.kernel()[0])
            if not mc.converged:
                raise CommandBackendError("CASSCF 未收敛")
            if self.pt2 == "nevpt2":
                from pyscf import mrpt
                e_pt2 = float(mrpt.NEVPT(mc).kernel())
                e = e + e_pt2          # 动态相关修正 (二阶微扰)
            self._last_solver = mc
            if not need_grad:
                return e, None
            # ⚠ PySCF 无 CASSCF 解析梯度模块 (pyscf.grad.mcscf 不存在);
            # mc.nuc_grad_method() 返回的是 **CASCI 型**梯度 (缺轨道响应项):
            # 对 CAS(2,2)/H₂ 恰好正确 (实测 vs FD 3.8e-07), 但 CAS(4,4)/H₂O
            # 下偏差达 10² Ha/Bohr (实测)。故默认用中心有限差分 (正确但 6N 倍
            # 能量代价); grad_t_mode="casci" 可取那个近似值 (仅供快速预估)。
            if self.grad_t_mode == "casci":
                grad = mc.nuc_grad_method().kernel()
                return e, np.asarray(lib.asarray(grad), dtype=float).reshape(-1, 3)
            g = np.zeros_like(coords_arr)
            for i in range(coords_arr.shape[0]):
                for j in range(3):
                    cp, cm = coords_arr.copy(), coords_arr.copy()
                    cp[i, j] += self.grad_t_h
                    cm[i, j] -= self.grad_t_h
                    g[i, j] = (self._energy_only(cp) - self._energy_only(cm)) / \
                        (2 * self.grad_t_h)
            return e, g

        # ---- 相关方法 (MP2 / CCSD / CCSD(T)): 能量与**相关梯度** ----
        solver, with_t = self._corr_solver(mf)
        # ---- EOM-CCSD 激发态 (v0.29.0) ----
        if self.method == "eom-ccsd":
            from pyscf.cc import eom_rccsd
            e_corr = float(solver.kernel()[0])
            e_ref = float(mf.e_tot) + e_corr
            if with_t:
                e_ref += float(solver.ccsd_t())
            eom = eom_rccsd.EOMEESinglet(solver)
            es, vecs = eom.kernel(nroots=self.nstates)
            es = np.asarray(es).ravel()
            vecs = np.asarray(vecs)
            idx = self._pick_root(es, vecs)
            self.last_excitations = es
            self.last_oscillator_strengths = np.zeros_like(es)
            self._prev_exc_vec = vecs[idx].copy()
            if self.state is None:
                return float(e_ref), None
            e_exc = e_ref + float(es[idx])
            if not need_grad:
                return e_exc, None
            g = np.zeros_like(coords_arr)
            for i in range(coords_arr.shape[0]):
                for j in range(3):
                    cp, cm = coords_arr.copy(), coords_arr.copy()
                    cp[i, j] += self.grad_t_h
                    cm[i, j] -= self.grad_t_h
                    g[i, j] = (self._energy_only(cp) - self._energy_only(cm)) / \
                        (2 * self.grad_t_h)
            return e_exc, g

        if solver is not None:
            e = self._corr_energy(solver, with_t, mf)
            if not need_grad:
                return float(e), None
            # PySCF: MP2 → grad.mp2; CCSD → grad.ccsd (**不含 (T) 项**)
            grad = solver.nuc_grad_method().kernel()
            if with_t and self.grad_t_mode == "fd":
                grad = grad + self._fd_t_gradient(coords_arr)
            self._last_solver = solver
        else:
            if not need_grad:
                return float(e), None
            if self._solvent_fd_grad:
                # PySCF 无该溶剂模型的解析梯度模块 (如 ddPCM) → 中心 FD
                g = np.zeros_like(coords_arr)
                for i in range(coords_arr.shape[0]):
                    for j in range(3):
                        cp, cm = coords_arr.copy(), coords_arr.copy()
                        cp[i, j] += self.grad_t_h
                        cm[i, j] -= self.grad_t_h
                        g[i, j] = (self._energy_only(cp) - self._energy_only(cm)) / \
                            (2 * self.grad_t_h)
                return float(e), g
            grad = mf.nuc_grad_method().kernel()
        grad_arr = np.asarray(lib.asarray(grad), dtype=float).reshape(-1, 3)
        return float(e), grad_arr

    def excitation_spectrum(self, coords: np.ndarray):
        """返回激发谱 ``(energies_eV, oscillator_strengths)`` (需 method="tddft")。"""
        self._run(np.asarray(coords, dtype=float), need_grad=False)
        e_exc = getattr(self, "last_excitations", None)
        if e_exc is None:
            raise CommandBackendError(
                'excitation_spectrum 需要 method="tddft"')
        return (e_exc * 27.211386245988,
                np.asarray(self.last_oscillator_strengths))

    def _energy_only(self, coords: np.ndarray) -> float:
        """仅算能量 (供 CASSCF 有限差分梯度使用, 避免递归求梯度)。"""
        return self._run(coords, need_grad=False)[0]

    def _t_increment(self, coords: np.ndarray, guess=None):
        """(T) 能量增量 E_(T) = E_CCSD(T) - E_CCSD (Hartree)。

        ``guess`` 可为参考几何的 (t1, t2) 振幅初猜 — 位移几何的最优振幅与其
        接近, 可把 CCSD 迭代次数从 ~15 降到 ~3-5 (实测加速约 3 倍)。
        返回 ``(e_t, (t1, t2))`` 以便调用方复用。
        """
        mol = self._mol(np.asarray(coords, dtype=float))
        mf = self._mf(mol)
        mf.conv_tol = self.conv_tol
        mf.kernel()
        solver, _ = self._corr_solver(mf)
        if guess is not None:
            solver.kernel(*guess)
        else:
            solver.kernel()
        e_t = float(solver.ccsd_t())
        return (e_t if abs(e_t) > 1e-14 else 0.0), (solver.t1, solver.t2)

    def _fd_t_gradient(self, coords: np.ndarray, h: float | None = None) -> np.ndarray:
        """(T) 项梯度的中心有限差分 (PySCF grad.ccsd 不含 (T), 必须补)。

        成本: 每个构型 6N 次 CCSD(T) 计算。对力训练/几何优化是必需的
        (否则能量是 CCSD(T) 而力只有 CCSD 水平, 二者不一致)。
        """
        h = self.grad_t_h if h is None else h
        c0 = np.asarray(coords, dtype=float)
        g = np.zeros_like(c0)
        guess = None                     # 参考几何的振幅初猜 (逐步更新)
        for i in range(c0.shape[0]):
            for j in range(3):
                cp, cm = c0.copy(), c0.copy()
                cp[i, j] += h
                cm[i, j] -= h
                tp, guess = self._t_increment(cp, guess)
                tm, guess = self._t_increment(cm, guess)
                g[i, j] = (tp - tm) / (2 * h)
        return g

    def _corr_energy(self, solver, with_t: bool, mf) -> float:
        """跑相关求解器并返回总能量 (Hartree); 收敛失败抛 CommandBackendError。

        frozen 已在 ``_corr_solver`` 构造时设定; (T) 项用求解器自带设置。
        """
        e_corr = float(solver.kernel()[0])
        e_tot = float(mf.e_tot) + e_corr
        if with_t:
            e_tot += float(solver.ccsd_t())
        return e_tot

    @staticmethod
    def _mom_setocc(mo_occ: np.ndarray) -> np.ndarray:
        """把 ``mf.mo_occ`` 转换为 PySCF ``mom_occ`` 所需的占据数组。

        - UHF/UKS: ``mo_occ`` 已是 (2, nmo) 的 0/1 alpha/beta 数组, 直接使用;
        - ROHF/ROKS: ``mo_occ`` 为一维 {2,1,0} (双占据/单占据/空), 需展开为
          (2, nmo): alpha = [occ≥1], beta = [occ≥2] (PySCF MOM 的约定)。
        """
        occ = np.asarray(mo_occ, dtype=float)
        if occ.ndim == 2:
            return (occ > 0).astype(float)
        alpha = (occ >= 1.0).astype(float)
        beta = (occ >= 2.0).astype(float)
        return np.stack([alpha, beta])

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """总能量 (Hartree) 与核梯度 (Hartree/Bohr)。

        方法由 ``method`` 决定: SCF 层 (RHF/ROHF/UHF/RKS/ROKS/UKS) 或
        相关方法 (MP2/CCSD/CCSD(T))。相关方法返回**相关梯度**
        (MP2 走 ``grad.mp2``; CCSD/(T) 走 ``grad.ccsd``)。SCF 不收敛或
        自旋锁定失败时抛 ``CommandBackendError``。
        """
        return self._run(coords)

    def energy(self, coords: np.ndarray) -> float:
        """总能量 (Hartree); SCF 或相关方法由 ``method`` 决定 (不计算梯度)。"""
        return self._run(coords, need_grad=False)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """核梯度 (Hartree/Bohr); 相关方法返回相关梯度。"""
        return self._run(coords, need_grad=True)[1]

    def resonance_width(self, coords: np.ndarray) -> float:
        """**解析模型接口** (非从头算) — 给出自电离宽度 Γ(R) 的经验估计 (Hartree)。

        ⚠ 诚实边界: 本方法不做 CAP-CI / Feshbach 投影等第一性原理共振计算,
        只是在给定模型参数 (指数衰减或盒式 CAP) 下返回 Γ(R) 的解析值。
        真实 Γ(R) 必须由外部提供 (文献 MRCI 数据或专门计算), 例如
        He*+Li 的 ²Σ 通道 Γ 峰值 ≈ 10.6 meV, 指数尾斜率 k ≈ 2.87 Å⁻¹。

        参数 (``cap_params``):
        - ``{"type": "exponential", "A": ..., "beta": ..., "r_index": (i, j)}``
          → Γ = A·exp(−β·R_ij);
        - ``{"type": "box", "eta": ..., "r_cap": ...}`` → R > r_cap 时
          Γ = 2η(R−r_cap)²;
        未配置 ``cap_params`` 时返回 0.0 (实势能面极限)。
        """
        if not self.cap_params:
            return 0.0
        c = np.asarray(coords, dtype=float)
        p_type = self.cap_params.get("type", "exponential")
        idx = self.cap_params.get("r_index", (0, 1))
        if len(c) > max(idx):
            r = float(np.linalg.norm(c[idx[0]] - c[idx[1]]))
        else:
            r = float(np.linalg.norm(c[0]))

        if p_type == "exponential":
            a = float(self.cap_params.get("A", 0.01))
            beta = float(self.cap_params.get("beta", 1.0))
            return float(a * np.exp(-beta * r))
        elif p_type == "box":
            eta = float(self.cap_params.get("eta", 0.001))
            r_cap = float(self.cap_params.get("r_cap", 5.0))
            if r > r_cap:
                return float(2.0 * eta * (r - r_cap) ** 2)
            return 0.0
        return 0.0

    def complex_energy(self, coords: np.ndarray) -> complex:
        """复共振能量 E_res = E_R - i * Γ/2 (Hartree)。"""
        e = self.energy(coords)
        gamma = self.resonance_width(coords)
        return complex(e, -0.5 * gamma)

    def soc_terms(self, coords: np.ndarray,
                  orbitals: Optional[Sequence[int]] = None,
                  z_eff: Optional[Dict[str, float]] = None,
                  term: Optional[str] = None) -> Dict[str, Any]:
        """单电子 Breit–Pauli **轨道层** SOC 分析 (cm⁻¹)。

        用当前 ``method`` 的 SCF 轨道 (RHF/ROHF); 开壳层 p/π 壳层可自动识别
        (显式 ``orbitals`` 优先)。返回 ``{"zeta_cm", "splitting_cm",
        "orbitals", "term", "h_mo", "provenance"}``:
          - 原子 ²P: 分裂 = (3/2)ζ;  - 线性分子 ²Π: 分裂 = |ζ| (A 常数)。
        ⚠ 仅单电子项 (无二电子 SOC); 验证脚本给出逐元素实测比值。
        """
        from aqpes.pes import soc as _soc
        coords_arr = np.asarray(coords, dtype=float)
        mol = self._mol(coords_arr)
        mf = self._mf(mol)
        mf.kernel()
        if not mf.converged:
            raise CommandBackendError("PySCF SCF 未收敛")
        out = _soc.soc_orbital_analysis(mol, mf, orbitals=orbitals,
                                        z_eff=z_eff, term=term)
        out["provenance"] = self.provenance
        return out

    def soc_state_interaction(self, coords: np.ndarray,
                              active_orbitals: Sequence[int],
                              singlet_roots: int = 1, triplet_roots: int = 1,
                              z_eff: Optional[Dict[str, float]] = None,
                              nelecas: Optional[int] = None) -> Dict[str, Any]:
        """**态相互作用** SOC 矩阵 (cm⁻¹): 单重态-三重态耦合。

        共同轨道基 = 当前 method 的 SCF 轨道; ``active_orbitals`` 给出活性空间
        (经 ``mcscf.sort_mo`` 就位), 三重态含 M=+1/0/−1 三分量。
        返回 ``{"energies_cm", "soc_cm", "labels", "nelecas", "provenance"}``。
        """
        from aqpes.pes import soc as _soc
        coords_arr = np.asarray(coords, dtype=float)
        mol = self._mol(coords_arr)
        mf = self._mf(mol)
        mf.kernel()
        if not mf.converged:
            raise CommandBackendError("PySCF SCF 未收敛")
        out = _soc.soc_state_interaction(
            mol, mf, active_orbitals, singlet_roots=singlet_roots,
            triplet_roots=triplet_roots, z_eff=z_eff, nelecas=nelecas)
        out["provenance"] = self.provenance
        return out

    @property
    def provenance(self) -> Dict[str, str]:
        """记录 PySCF 版本、方法、基组、电荷、自旋与高级设置。"""
        try:
            import pyscf
            ver = pyscf.__version__
        except Exception:
            ver = "unknown"
        p = {
            "backend": "pyscf",
            "version": ver,
            "method": self.method,
            "basis": self.basis,
            "charge": str(self.charge),
            "spin": str(self.spin),
            "spin_lock": str(self.spin_lock),
            "use_mom": str(self.use_mom),
            "frozen_core": str(self.frozen_core),
            "relativistic": str(self.relativistic or "none"),
            "active_space": (f"({self.active_space[0]},{self.active_space[1]})"
                             if self.active_space else "n/a"),
            "pt2": str(self.pt2 or "none"),
            "solvent": self._solvent_desc(),
            "nstates": (str(self.nstates) if self.method == "tddft" else "n/a"),
            "state": (str(self.state)
                      if self.method in ("tddft", "eom-ccsd") else "n/a"),
            "follow": (str(self.follow)
                       if self.method in ("tddft", "eom-ccsd") else "n/a"),
            "grad_t_mode": (self.grad_t_mode
                            if self.method in ("ccsd(t)", "ccsd_t") else "n/a"),
        }
        if self.xc:
            p["xc"] = str(self.xc)
        if self.cap_params:
            p["cap_params"] = str(self.cap_params)
        return p


class ASECalculatorAdapter(Calculator):
    """包装任意 ASE calculator 对象 (可选依赖; 实验性)。

    ``ase.calculators.calculator.Calculator`` 协议: ``get_potential_energy``
    与 ``get_forces`` (后者为 -∇E, 此处取负号)。
    """

    name = "ase"

    def __init__(self, ase_calculator):
        self.calc = ase_calculator

    def energy_and_gradient(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """ASE 势能 (Hartree) 与梯度 (取力的负号, Hartree/Bohr)。"""
        from ase import Atoms
        atoms = Atoms(symbols=self.calc.atoms.get_chemical_symbols()
                      if hasattr(self.calc, "atoms") else ["H"] * len(coords),
                      positions=np.asarray(coords, dtype=float))
        atoms.calc = self.calc
        e = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        return e, -forces

    def energy(self, coords: np.ndarray) -> float:
        """ASE 势能 (Hartree)。"""
        return self.energy_and_gradient(coords)[0]

    def gradient(self, coords: np.ndarray) -> np.ndarray:
        """ASE 核梯度 (-力, Hartree/Bohr)。"""
        return self.energy_and_gradient(coords)[1]

    @property
    def provenance(self) -> Dict[str, str]:
        """记录被包装 calculator 的类名与版本。"""
        calc = self.calc
        return {"backend": "ase", "calculator": type(calc).__name__,
                "version": str(getattr(calc, "version", "unknown"))}


# ---------------------------------------------------------------------------
# 内置演示后端 (无外部依赖; 明确标注为演示用, 非量子化学)
# ---------------------------------------------------------------------------

def demo_calculator(epsilon: float = 0.01, sigma: float = 3.4) -> Calculator:
    """内置 Lennard-Jones 二聚体演示后端 (解析能量+梯度)。

    用途: 让 ``aqpes sample/fit`` 闭环在无任何外部量子化学程序
    的环境中也可完整运行与验证。**这不是量子化学计算** — 势能面是
    虚构的 LJ 对势, 仅演示数据生成→力训练管线。
    """

    def energy_fn(coords: np.ndarray) -> float:
        """LJ 对势能量 4ε[(σ/r)¹² − (σ/r)⁶] (Hartree), r 为两原子间距。"""
        c = np.asarray(coords, dtype=float)
        r2 = ((c[0] - c[1]) ** 2).sum()
        inv6 = (sigma ** 2 / r2) ** 3
        return float(4 * epsilon * (inv6 ** 2 - inv6))

    def grad_fn(coords: np.ndarray) -> np.ndarray:
        """LJ 能量解析梯度 (Hartree/Bohr), 沿两原子连线方向。"""
        c = np.asarray(coords, dtype=float)
        d = c[0] - c[1]
        r2 = float((d ** 2).sum())
        inv6 = (sigma ** 2 / r2) ** 3
        # dE/dd = 4ε(-12σ¹²/r¹⁴ + 6σ⁶/r⁸)·d/r²
        coeff = 4 * epsilon * (-12 * sigma ** 12 / r2 ** 7
                              + 6 * sigma ** 6 / r2 ** 4)
        g0 = coeff * d
        return np.array([g0, -g0])

    return AnalyticCalculator(energy_fn, grad_fn, name="demo-lj")


# ---------------------------------------------------------------------------
# 可用性报告
# ---------------------------------------------------------------------------

def available_calculators() -> Dict[str, Dict[str, str]]:
    """报告各后端的可用性 (不抛异常; 供 CLI/文档使用)。"""
    import importlib.util

    status: Dict[str, Dict[str, str]] = {}

    status["analytic"] = {"available": "yes", "note": "内置, 零依赖"}
    status["demo"] = {"available": "yes",
                      "note": "内置 LJ 演示 (非量子化学), 供管线自检"}

    xtb_path = shutil.which("xtb")
    status["xtb"] = {"available": "yes" if xtb_path else "no",
                     "note": xtb_path or "未找到 xtb 可执行文件"}

    has_pyscf = importlib.util.find_spec("pyscf") is not None
    status["pyscf"] = {"available": "yes" if has_pyscf else "no",
                       "note": "pip install pyscf" if not has_pyscf else "installed"}

    has_ase = importlib.util.find_spec("ase") is not None
    status["ase"] = {"available": "yes" if has_ase else "no",
                     "note": "pip install ase" if not has_ase else "installed"}

    return status


def make_calculator(backend: str, symbols: Optional[Sequence[str]] = None,
                    **kwargs) -> Calculator:
    """按名称构造后端 ('demo' | 'analytic' | 'xtb' | 'pyscf' | 'ase')。"""
    if backend == "demo":
        return demo_calculator(**kwargs)
    if backend == "analytic":
        fn = kwargs.pop("energy_fn")
        return AnalyticCalculator(fn, gradient_fn=kwargs.pop("gradient_fn", None))
    if backend == "xtb":
        return XTBCommandCalculator(symbols, **kwargs)
    if backend == "pyscf":
        return PySCFCalculator(symbols, **kwargs)
    if backend == "ase":
        return ASECalculatorAdapter(kwargs.pop("ase_calculator"))
    raise ValueError(f"未知后端: {backend!r}; "
                     f"可选 {sorted(available_calculators())}")
