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
| `cluster_solvent_validation_1558906.log` | 1558906 (xc003, 278 s) | PCM / ddPCM / SMD 溶剂模型 | 全部通过 (8/8)：ε→1 = 0；介电单调；跨模型比 1.49；PCM 梯度 vs FD **4.4e-07**；SMD CH₄ **+2.19 vs 实验 +1.95** |

## 自旋-轨道耦合 (SOC) 验证状态

日志归档: `validation/cluster_soc_validation_1558957.log`（Slurm 1558957, xc003, 17 s,
**存在失败项** —— 失败组与原因逐条列在下方, 不做选择性呈现）。
`scripts/validate_soc.py`（主仓库中的同一脚本）逐组判据与**当前**实测状态：

| 组 | 判据 | 状态 |
|---|---|---|
| A | 类氢离子 ζ = α²Z⁴/48（由 ⟨r⁻³⟩=Z³/24 精确推出） | **通过**：n_p=18 时比值 **0.99997**；²P 分裂 5.84348 vs 精确 5.84366 cm⁻¹ |
| B | 原子 ²P 精细结构 vs 实验 (F 404.14 / Cl 882.36 / Br 3685.3 cm⁻¹) | **通过**（判据为量级+系统性）：单电子 BP 比 **1.463 / 1.142 / 1.001** → 随 Z 逼近实验（二电子项缺失的系统行为） |
| C | 原点平移不变性 + 全局旋转不变性 | 平移 **8e-17** ✓；旋转检验因耦合为零而失效（与 E 同源，见下） |
| D | C₂ᵥ 对称性选择定则 (³B₁–¹A₁ 只有 B₁ 分量) | z 分量 ≈ 2e-21 ✓（对称性禁阻正确）；**非零分量未复现 → 与 E 同源** |
| E | OH ²Π：轨道层 ζ vs CI 层 SOC 交叉验证 + 实验 A=139.2 cm⁻¹ | 轨道层 **ζ_π = 254.29 cm⁻¹**（比实验 1.83，与 F 的轻元素偏大一致）✓；A(R) 扫描 ✓；**CI 层耦合 = 0 ✗（排查中）** |
| F | 自旋纯度 (fix_spin_) 与 Hermitian 性 | **通过**：⟨S²⟩ 0.000000/2.000000；反对称偏差 0.00e+00 |
| G1 | 跃迁密度 vs PySCF 参考实现 (make_rdm1s/trans_rdm1s) | **通过**：相对差 **0e+00 / 2.1e-16**（行列式代数与 CI 布局定址均正确） |

**结论的诚实表述**：SOC 的**轨道层**（ζ/精细结构分裂、A(R) 扫描）已通过精确类氢
标定、原子实验对比、平移不变性与单元测试；**态相互作用层**（CI 波函数间的
单-三态耦合矩阵）目前仍在排查一个使耦合为零的缺陷，**尚未宣称完成**。
相关证据（含失败组）一并归档于本节，不做选择性呈现。

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
