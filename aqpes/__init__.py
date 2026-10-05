"""AutoQuantumPES — 势能面 (PES) 从头算计算与机器学习拟合工具包。

从 AutoQuantumDynamics 单体仓库拆分的**独立子集**:
  - 电子结构后端: PySCF (SCF/DFT/MP2/CCSD(T)/CASSCF/NEVPT2/TD-DFT/EOM-CCSD/
    ddCOSMO/PCM/ddPCM/SMD/X2C/SOC), xTB, ASE, 内置演示势
  - 工作流: 几何优化, 内坐标/松弛扫描, 谐振频率, CI-NEB 过渡态, IRC 反应路径
  - 激发态势能面 (TD-DFT/EOM-CCSD), 自旋-轨道耦合 (单电子 Breit-Pauli)
  - 隐式溶剂 (ddCOSMO / PCM / ddPCM / SMD)
  - 数据集与主动学习: AbInitioData, 对称函数, 神经网络集成拟合
  - CLI: aqpes sample/fit/opt/scan/freq/soc
"""
__version__ = "0.31.0"
