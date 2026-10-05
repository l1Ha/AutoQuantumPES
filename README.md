# AutoQuantumPES — 势能面计算与拟合工具包

势能面 (PES) 从头算计算与机器学习拟合工具包：电子结构 → 势能面工作流 →
数据集/主动学习 → 神经网络拟合，全部在同一个 API 下完成，并可用 CLI 驱动。

**自包含、零强制外部依赖**（只需 numpy；PySCF / xTB / ASE 为可选后端），
`pip install -e .` 即可独立使用。

```bash
pip install -e .            # 安装 (aqpes 包 + aqpes 命令)
pip install -e ".[pyscf]"   # 电子结构后端 (PySCF)
aqpes sample --backend pyscf --input geom.xyz --method ccsd(t) --basis cc-pvdz -o data.npz
aqpes fit --data data.npz -o model.pkl
```

## 目录结构

```text
AutoQuantumPES/
├── aqpes/
│   ├── cli.py          # 命令行: info/sample/fit/backends/opt/scan/freq/soc
│   ├── core/           # 基础: 体系/势能面容器、周期表常量、验证与可复现性工件
│   ├── pes/            # 势能面核心
│   │   ├── calculators.py  # PySCF/xTB/ASE/演示 后端 (约 20 种电子结构方法)
│   │   ├── abinitio.py     # 数据集容器 + 几何采样 (含主动学习接口)
│   │   ├── optimize.py     # BFGS 几何优化 / 数值 Hessian / 谐振频率
│   │   ├── scan.py         # 内坐标扫描 / 线性路径 / 松弛(约束)扫描
│   │   ├── neb.py          # CI-NEB 过渡态 / IRC 反应路径
│   │   ├── soc.py          # 自旋-轨道耦合 (单电子 Breit–Pauli, 态相互作用)
│   │   ├── analytic.py     # 解析面 (Morse/Harmonic/LJ)
│   │   ├── leps.py         # LEPS 面 (含解析鞍点, 用于 NEB/IRC 校验)
│   │   ├── eckart.py       # Eckart 垒
│   │   └── builder.py      # PES 装配
│   └── nn/             # 机器学习拟合
│       ├── model.py        # 前馈 NN (含 GPU/float32 路径)
│       ├── symmetry.py     # 对称函数/置换不变特征
│       ├── ensemble.py     # 委员会 (不确定性)
│       ├── train.py        # 训练器 (含早停/学习率调度)
│       ├── active_learning.py  # 主动学习闭环
│       └── dataset.py      # 数据集
├── tests/              # 90 项独立单元测试 (pytest/unittest, 无需 PySCF)
├── validation/         # 服务器 (Slurm) 验证日志归档 —— 索引见 VALIDATION.md
└── tools/sync_from_monorepo.sh   # 与主仓库同步 (拆分的来源与再生成)
```

## 能力与验证证据

所有证据来自**集群真实计算**(Slurm 作业日志归档在 `validation/`), 不是文献抄录。
判据多为**独立物理约束**: FCI 精确对照、解析鞍点、精确 Dirac 能、ε→1 极限、
对称性选择定则、vs 有限差分等。

| 能力 | 关键验证 (服务器 Slurm) |
|---|---|
| SCF/DFT (RHF/ROHF/UHF/RKS/ROKS/UKS) | 能量 vs 文献 < 0.12 mHa (`validate_correlated_methods`) |
| MP2 / CCSD / CCSD(T) + **解析梯度** | 梯度 vs 有限差分 **3.5e-07** Ha/Bohr; CCSD ≡ FCI (H₂ 0.0000 mHa) |
| 几何优化 / 扫描 / 谐振频率 | H₂/CCSD(T)/cc-pVQZ **0.7417 Å** (= 文献); H₂O 频率偏差 **1.86%**, 0 虚频 |
| CI-NEB 过渡态 | LEPS 势垒误差 **0.00%**; H₃ TS **恰 1 虚频 −1465 cm⁻¹** |
| IRC 反应路径 | Ishida–Morokuma 平均梯度; 双方向**完全一致**, 100% 单调 |
| ECP / 赝势 | AuH/cc-pVDZ-PP 梯度 vs FD **4.5e-07** |
| 标量相对论 (X2C) | vs **精确 Dirac 能**: H 0.4%, He⁺ 1.0% |
| CASSCF 多参考 | H₂ 解离 R=4 Bohr: \|RHF−FCI\| 105.6 mHa → \|CASSCF−FCI\| **0.9 mHa** |
| **AVAS 自动活性空间** | 由 AO 标签自动构造活性空间; 闭壳层恒等式 **8.5e-14**; N₂ 相关 0.132 Ha; AVAS+CASSCF/态平均/SCI/NEVPT2 全部可叠加; 标签格式实测标定 (列表 ✓ / `;`,`,` ✗ 静默 ncas=0) (Slurm 1559297, 14 s, exit 0) |
| **复合方法 (CBS + CCSD(T) 加和)** | 三点指数 HF 外推重构误差 **0**; H₂ **R_e = 0.7414 Å (偏差 0.003%)**; H₂O 的 HF/相关两分量 CBS 都优于 cc-pVQZ; 对 CBS 极限偏差 −1.05 mHa (5Z 本身 +0.25 mHa, 变分自洽); **过冲边界如实记录** (Slurm 1559235, 79 s, exit 0) |
| **CASCI + 选择组态 CI** (大活性空间) | SCI ≡ **稠密 FCI** (CAS(6,6) **1.44e-12**, CAS(8,8) **4.83e-13 Ha**); **CAS(14,14)/cc-pVDZ 稠密 11,778,624 维 → 9 s**; 变分单调收敛 (末两档 0.003 mHa); R_e: CASSCF(8,8) 1.1220 vs CASCI-SCI(14,14) 1.1215 Å (一致 0.0005 Å) (Slurm 1559215, 133 s, exit 0) |
| **态平均 CASSCF 激发态** + NEVPT2(root) | SA(2) vs **FCI 单重态 8.9e-16**; LiH 避交叉扫描: 无交叉、CI 向量重叠 **0.999895**、ΔE_min 1.5835 eV @ 5.60 Bohr; NEVPT2 对 FCI 拉近 **25.0→9.3 mHa** (Slurm 1559180, 514 s, exit 0) |
| NEVPT2 动态相关 | H₂/CAS(8,2) 距 FCI **0.10 mHa**; LiH 改善 2.5–2.7× |
| TD-DFT/TDHF 激发态势能面 | H₂O 最低激发 **7.605 eV vs 实验 7.4 (2.8%)**; 振子强度 Σf < TRK 上界 |
| EOM-CCSD 激发态 | vs FCI 单重态激发能 **偏差 0.000 eV**; H₂O 8.675 vs TD-DFT 8.061 eV |
| 隐式溶剂 ddCOSMO | ε→1 极限 **0.000000**; Li⁺ Born 比值 1.19; 梯度 vs FD 3.6e-07 |
| PCM / ddPCM / SMD | 8/8 组通过: ε→1 = 0、介电单调、跨模型比 1.49、PCM 梯度 vs FD **4.4e-07**、SMD 非静电项 (CH₄/water **+2.19 vs 实验 +1.95**) |
| 自旋-轨道耦合 (SOC) | 类氢精确标定 ζ = α²Z⁴/48 → 比 **0.99997**; 原子 ²P 分裂 vs 实验 F/Cl/Br = **1.457 / 1.117 / 0.996**; 平移 4.8e-6 cm⁻¹ / 旋转 8.5e-14; C₂ᵥ 选择定则与 Wigner–Eckart 严格成立; OH ²Π 轨道层≡CI 层 (**6.3e-16**) vs 实验 A 比 1.63; 跃迁密度 vs PySCF **2.1e-16** |
| NN 拟合 / 主动学习 | 81 点 MP2 训练集留出集 RMSE **0.178% of span** (判据 <1%) |
| 集群批量生产 | Slurm 阵列 + 动态分片 (`production.py` 在主仓库) |

详见 [VALIDATION.md](VALIDATION.md) 的逐条日志索引。

## Python API 速览

```python
import numpy as np
from aqpes.pes.calculators import make_calculator
from aqpes.pes.optimize import optimize_geometry, harmonic_frequencies
from aqpes.pes.scan import scan_bond
from aqpes.pes.neb import neb_path, irc_path
from aqpes.pes.abinitio import AbInitioData
from aqpes.nn import NNTrainer, TrainingConfig

# 1) 电子结构后端 (默认单位 Bohr)
calc = make_calculator("pyscf", symbols=["O", "H", "H"], basis="cc-pvdz",
                       method="ccsd(t)", frozen_core=True)
E, g = calc.energy_and_gradient(coords)

# 2) 势能面工作流
opt = optimize_geometry(calc, coords)               # BFGS + 解析梯度
freq = harmonic_frequencies(calc, opt.coords)       # 数值 Hessian + 投影
ts = neb_path(calc, reactant, product)              # CI-NEB (+climbing image)
irc = irc_path(calc, ts.coords, mode)               # 双向 IRC

# 3) 激发态 / 溶剂 / SOC
exc = make_calculator("pyscf", symbols=["O","H","H"], method="tddft",
                      xc="b3lyp", nstates=4, state=1, solvent="water")
soc = make_calculator("pyscf", symbols=["O","H"], method="rohf", spin=1,
                      basis="cc-pvtz")
info = soc.soc_terms(soc_geom)                      # ζ / 精细结构分裂

# 4) 数据集 → ML 拟合
data = AbInitioData.sample_geometries(calc, coords, ranges=((-0.4, 0.4),),
                                      n_per_dim=5)
trainer = NNTrainer(data, TrainingConfig(epochs=2000))
model = trainer.fit()
```

## CLI

```text
aqpes info                                  # 环境与版本
aqpes backends                              # 后端可用性 (pyscf/xtb/ase/demo)
aqpes sample   --backend pyscf --input g.xyz --method mp2 ... -o data.npz
aqpes fit      --data data.npz -o model.pkl [--committee 4]
aqpes opt      --input g.xyz --method ccsd(t) --basis cc-pvdz
aqpes scan     --input g.xyz --mode bond --range 1.2 3.0 --n 15
aqpes freq     --input g.xyz --method mp2
aqpes soc      --input g.xyz --method rohf --basis cc-pvtz [--orbitals 3 4]
               [--active-orbitals 3 4 --singlet-roots 2 --triplet-roots 2]
```

`sample/opt/scan/freq` 共通的后端参数：`--charge/--spin/--method/--basis/--xc/
--frozen-core/--solvent*/--solvent-model/--pcm-variant`。

## 诚实边界 (与商业软件的差距, 逐条有实测依据)

- **SOC 仅单电子 Breit–Pauli**：无二电子 SOC/屏蔽项 → 轻元素系统性偏大
  (F 1.46×、Cl 1.14×)，重元素接近实验 (Br 1.00×)。可用 `z_eff` 做经验屏蔽修正，
  但须标注为经验。
- **无 MRCI/RASPT2**：变分式多参考 CI 未实现 (CASSCF+NEVPT2 是当前最强的多参考路径)。
- **无 CCSD(T)-F12 / 显式相关**。
- **梯度粒度**：MP2/CCSD/CCSD(T) 有解析梯度；CASSCF/NEVPT2/TD-DFT(按态)/EOM-CCSD
  为有限差分 (PySCF 无相应解析梯度模块，已实测确认)。
- **SMD 为 PySCF 实验性实现**：H₂O 绝对值与实验差 ~2.5 kcal/mol，只作数量级核查；
  **ddPCM** 在 PySCF 中标注 *under testing* (ε=1 内部除零，梯度需 FD)。
- **根跟踪为实验性**：重叠判据在 H₂ 态交叉窗口未改善连续性 (负结果已如实记录)。
- **无 G4/W1 复合方法**。
- 显式溶剂 (QM/MM 微溶剂化) 未实现。

## 开发与同步

本包的内核代码与上游自研电子结构项目同源；可用脚本从上游仓库重新生成 PES
子集（import 前缀改写 + 剔除上游专属模块，README/VALIDATION/pyproject 等
本包自有文件不会被覆盖）：

```bash
bash tools/sync_from_upstream.sh /path/to/upstream-repo
```

- 版权与许可：见 [LICENSE](LICENSE)。
- 版本与变更：见 `aqpes.__version__` 与 [VALIDATION.md](VALIDATION.md)（每次
  服务器验证后的发布记录）。
- 仓库：<https://github.com/l1Ha/AutoQuantumPES>
