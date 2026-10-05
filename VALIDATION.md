# 验证证据索引 (服务器 Slurm)

本目录归档了 AutoQuantumPES 所覆盖能力的**集群真实计算日志**。判据原则：
优先使用**独立物理约束**（FCI 精确值、解析鞍点、精确 Dirac 能、ε→1 极限、
对称性选择定则、vs 有限差分），而不是文献数值的记忆比较。

| 日志 | Slurm | 覆盖能力 | 结果要点 |
|---|---|---|---|
| `cluster_correlated_validation_1558297.log` | 1558297 (xc002, 1097 s) | MP2/CCSD/CCSD(T) 与解析梯度 | 全部通过：vs 文献 < 0.12 mHa；CCSD ≡ FCI (H₂ 0.0000 mHa)；梯度 vs FD **3.5e-07** |
| `cluster_pes_workflow_validation_1558433.log` | 1558433 (xc002, 800 s) | 几何优化/扫描/频率/ECP/端到端拟合 | 全部通过：H₂/CCSD(T)/cc-pVQZ **0.7417 Å**；H₂O 频率 1.86%；留出集 RMSE **0.178% of span** |
| `cluster_transition_state_validation_1558518.log` | 1558518 (xc016, 2344 s) | CI-NEB 过渡态 | 全部通过：LEPS 势垒误差 **0.00%**；H₃ TS **恰 1 虚频 −1465 cm⁻¹** |
| `cluster_irc_x2c_validation_1558653.log` | 1558653 (xc016, 1476 s) | IRC 反应路径 (+X2C 附带项) | IRC 全部通过：双方向**完全一致**、100% 单调、TS 首步下降；**X2C 组的"精确 Dirac 参考值"脚本自身有 bug（打印出溢出数）+ 一个测试配置错误 → 见下方专项重验** |
| `cluster_x2c_dirac_validation_1558690.log` | 1558690 (xc002, 94 s) | X2C 标量相对论 (专项重验) | 全部通过：vs **精确 Dirac 能** H −6.63 vs −6.66 µHa (0.4%)、He⁺ 1.0% |
| `cluster_casscf_validation_1558715.log` | 1558715 (xc002, 255 s) | CASSCF 多参考 | **存在失败项，且为真实发现**：① 解离区判据阈值过紧 (实测误差 0.88 mHa，判据要求更小 → 非物理失败)；② **`mc.nuc_grad_method()` 的"解析"梯度 vs FD 差 1.23e+02 Ha/Bohr** → 证实 PySCF 该梯度为 CASCI 型（缺轨道响应），故实现改为有限差分（后续 `validate_casscf_grad_xcheck` 复核 FD 路径） |
| `cluster_casscf_grad_diagnostic_1558721.log` | 1558721 | CASSCF 梯度诊断 | 定位 ①② 的证据链（能量 vs FD、不同活性空间下的偏差） |
| `cluster_casscf_grad_xcheck_1558753.log` | 1558753 (xc003, 302 s) | CASSCF FD 梯度交叉验证 | 全部通过：两条独立路径同 R 一致性 **0.005%** |
| `cluster_nevpt2_validation_1558765.log` | 1558765 (xc003, 632 s) | NEVPT2 动态相关 | 全部通过：H₂/CAS(8,2) 距 FCI **0.10 mHa**；LiH 改善 2.5–2.7× |
| `cluster_tddft_validation_1558827.log` | 1558827 (xc003, 73 s) | TD-DFT/TDHF 激发态势能面 | 全部通过：H₂O **7.605 eV vs 实验 7.4 (2.8%)**；振子强度 Σf < TRK 上界 |
| `cluster_ddcosmo_validation_1558838.log` | 1558838 (xc003, 71 s) | ddCOSMO 隐式溶剂 | 全部通过 (7/7)：ε→1 **0.000000**；Li⁺ Born 比值 1.19；梯度 vs FD 3.6e-07 |
| `cluster_eom_ccsd_validation_1558875.log` | 1558875 (xc003, 668 s) | EOM-CCSD 激发态 + 根跟踪 | 全部通过：单重态激发能 vs FCI **0.000 eV**；**根跟踪为负结果**（重叠判据未改善连续性，已如实记录） |
| `cluster_composite_validation_1559235.log` | 1559235 (xc003, 79 s) | **复合方法 (CBS 外推 + CCSD(T) 加和)** | 全部通过 (A–D): 公式精确重构 **0 / 2.2e-16**; 外推质量 (vs 5Z, H₂O) HF **9.4e-4**、相关 **6.9e-3** (均优于 QZ); H₂ **R_e 0.7414 Å (0.003%)**; \|δ\|/\|相关\| 0.045; 对 CBS 极限 −1.1744757 Ha 偏差 −1.05 mHa (**过冲边界如实记录**, (QZ,5Z) 降到 −0.90) |
| `cluster_selected_ci_validation_1559215.log` | 1559215 (xc003, 133 s) | **选择组态 CI (大活性空间 CASCI)** | 全部通过 (A–E): SCI ≡ 稠密 FCI (**1.44e-12 / 4.83e-13 Ha**); **CASCI(14,14)/cc-pVDZ (11,778,624 维) 9 s**; 变分单调 (末两档 **0.003 mHa**); R_e 1.1220 (CASSCF(8,8)) vs **1.1215 Å** (CASCI-SCI(14,14)) 一致 0.0005 Å; 记录 CASSCF+SCI 的 RDM 接口不兼容并给出明确报错 |
| `cluster_sa_casscf_validation_1559180.log` | 1559180 (xc003, 514 s) | **态平均 CASSCF 激发态 + NEVPT2(root)** | 全部通过 (A–E): SA(2) 自旋纯 = **FCI 单重态** (8.9e-16/3.3e-16); LiH 避交叉扫描无交叉、CI 重叠 **0.999895**、ΔE_min 1.5835 eV @ 5.60 Bohr; NEVPT2 对 FCI 拉近 **25.04→9.32 mHa** (如实报告态平均轨道代价 8–10 mHa); 基态 = 单态 = FCI (0.0e+00) |
| `cluster_solvent_validation_1558906.log` | 1558906 (xc003, 278 s) | PCM / ddPCM / SMD 溶剂模型 | 全部通过 (8/8)：ε→1 = 0；介电单调；跨模型比 1.49；PCM 梯度 vs FD **4.4e-07**；SMD CH₄ **+2.19 vs 实验 +1.95** |

## 自旋-轨道耦合 (SOC) 验证状态 — **已完成 (A–G 全绿)**

日志归档: `validation/cluster_soc_validation_1559052.log`（Slurm 1559052, xc003, 19 s, exit 0）。

| 组 | 判据 | 结果 |
|---|---|---|
| A | 类氢离子 ζ = α²Z⁴/48（精确解析值） | **通过**: n_p=18 时比值 **0.99997**；²P 分裂 5.84348 vs 精确 5.84366 cm⁻¹ |
| B | 原子 ²P 精细结构 vs 实验 (F 404.14 / Cl 882.36 / Br 3685.3 cm⁻¹) | **通过**: 单电子 BP 比 **1.457 / 1.117 / 0.996**（随 Z 逼近实验，二电子项缺失的系统行为） |
| C | 原点平移不变性 + 全局旋转不变性 | **通过**: 平移 4.8e-6 cm⁻¹（相对 3.8e-7，受 SCF/CI 1e-10 收敛限）；旋转后耦合矢量模长相对差 8.5e-14 |
| D | C₂ᵥ 对称性选择定则（CH₂ ³B₂–¹A₁，分子在 xz 平面 → 只允许 R_x） | **通过**: 积分层只有 B₂ 分量（h_y, h_z ~1e-19）；态层 \|c(M=±1)\| **严格等量**（12.601551 = 12.601551）、\|c(M=0)\| = 5.8e-15（A₂ 禁阻） |
| E | OH ²Π：轨道层 ζ vs CI 层耦合交叉验证 + 实验 A=139.2 cm⁻¹ | **通过**: 226.1803 = 226.1803（相对差 **6.3e-16**）；vs 实验比 1.63；A(R) 225→228 cm⁻¹ |
| F | 自旋纯度 (fix_spin_ + ⟨S²⟩ 过滤) 与 Hermitian 性 | **通过**: ⟨S²⟩ 0.000000/2.000000；反对称偏差 0.00e+00 |
| G | 跃迁密度 vs PySCF 参考 + CI 布局恒等式 + Wigner–Eckart | **通过**: 密度 vs `make_rdm1s/trans_rdm1s` **2.1e-16**；ΣD_aa=2, ΣD_bb=1；\|c(+1)\|=\|c(−1)\| 严格相等 |

**验证期间发现并修复的两处真实根因**（均由服务器证据定位，非事后补记）:

1. **缺 −i 因子使态相互作用 SOC 恒为零**: libcint 的 `int1e_prinvxp` 返回
   `(r×∇)/r³` 的**实**矩阵，而物理角动量 `l = −i(r×∇)` → 矩阵变反 Hermitian，
   末端的 Hermitian 对称化把矩阵元**完全抵消**。轨道层 ζ 只取 |本征值|，
   对该相位因子不敏感 → 长期未暴露。
2. **`with_common_origin` 对 `int1e_prinvxp` 无效**: 该积分的 1/r³ 起点由
   **rinv 原点**控制 → 结果随分子平移漂移（实测 He⁺ 平移后 ζ 由 3.874 变
   0.005 cm⁻¹）。改用 `with_rinv_at_nucleus(ia)` 后平移不变性达数值噪声级。

**诚实边界**: 仅**单电子** Breit–Pauli（无二电子 SOC/屏蔽）→ 轻元素系统性偏大
（F 1.46×、Cl 1.12×），重元素接近实验（Br 1.00×）；提供 `z_eff` 经验修正选项。

## 复现

```bash
# 集群 (Slurm) —— 主仓库的同一脚本
sbatch scripts/sbatch_soc.sbatch            # SOC
sbatch scripts/sbatch_solvent.sbatch        # PCM/SMD
sbatch scripts/sbatch_eom.sbatch            # EOM-CCSD
...
# 本地 (无需 PySCF)
python -m unittest discover -s tests        # 90 项
```
