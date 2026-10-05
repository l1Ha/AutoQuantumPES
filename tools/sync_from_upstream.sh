#!/usr/bin/env bash
# 从**上游仓库**重新生成本独立包的 PES 子集 (可复现同步脚本)。
#
# 用法:  bash tools/sync_from_upstream.sh <上游仓库路径>
# 说明:
#   1) 同步 pes/ + nn/ + core 子集 + CLI，并把 import 前缀 autoquantum. 改写为 aqpes.;
#   2) 裁剪上游专属内容 (engine / run 命令 / 传播诊断 / DynamicsResult /
#      nn_pes_2d 等)，使本包只保留势能面计算与拟合;
#   3) 本脚本 **不覆盖** README.md / VALIDATION.md / pyproject.toml /
#      tools/ / validation/ —— 这些是本包自有文件。
set -euo pipefail
SRC="${1:?用法: bash tools/sync_from_upstream.sh <上游仓库路径>}"
DST="$(cd "$(dirname "$0")/.." && pwd)"
PKG=aqpes
[ -d "$SRC/autoquantum/pes" ] || { echo "找不到 $SRC/autoquantum/pes"; exit 1; }

rm -rf "$DST/$PKG/pes" "$DST/$PKG/nn" "$DST/$PKG/cli.py"
mkdir -p "$DST/$PKG/pes" "$DST/$PKG/nn" "$DST/$PKG/core" "$DST/tests"
cp "$SRC"/autoquantum/pes/*.py "$DST/$PKG/pes/"
cp "$SRC"/autoquantum/nn/*.py  "$DST/$PKG/nn/"
for f in base.py periodic.py validation.py; do cp "$SRC/autoquantum/core/$f" "$DST/$PKG/core/"; done
cp "$SRC/autoquantum/cli.py" "$DST/$PKG/cli.py"
for name in test_calculators test_pes test_leps test_abinitio test_optimize_scan \
            test_nn test_symmetry test_soc test_data_pipeline; do
  [ -f "$SRC/tests/$name.py" ] && cp "$SRC/tests/$name.py" "$DST/tests/$name.py"
done

python3 - "$DST" "$PKG" <<'PYEOF'
import pathlib
import re
import sys

dst, pkg = sys.argv[1], sys.argv[2]
root = pathlib.Path(dst)


def rw(text):
    return (text.replace("from autoquantum.", f"from {pkg}.")
                .replace("import autoquantum.", f"import {pkg}.")
                .replace("from autoquantum import", f"from {pkg} import"))


# ---- 1) import 前缀改写 (包内 + 测试) ----
for p in list(root.glob(f"{pkg}/**/*.py")) + list(root.glob("tests/*.py")):
    p.write_text(rw(p.read_text(encoding="utf-8")), encoding="utf-8")

# ---- 2) CLI: 去掉上游专属 engine / run 命令, 并改写说明与 info 内容 ----
cli = root / pkg / "cli.py"
t = cli.read_text(encoding="utf-8")
t = "".join(l for l in t.splitlines(True)
             if "core.engine import AutoPipeline" not in l)   # 前缀无关
t = re.sub(r"    run_parser = sub\.add_parser\(.*?\n\n    info_parser",
           "    info_parser", t, flags=re.S)
t = t.replace(
    '    if args.command == "run":\n        return _run_pipeline(args)\n    elif args.command == "info":',
    '    if args.command == "info":')
t = re.sub(r"\ndef _run_pipeline\(args\):.*?\n\ndef _read_xyz",
           "\n\ndef _read_xyz", t, flags=re.S)
t = t.replace("AutoQuantum: 分子反应动力学全维量子动力学自动计算平台",
              "AutoQuantumPES: 势能面计算与拟合 (PES 子集)")
t = t.replace('''        "available_dynamics": ["1d_scattering", "2d_wavepacket",
                               "1d_wavepacket", "2d_scattering_experimental"],
        "description": {
            "H2_1D": "H₂ 分子一维散射 (Morse 势)",
            "H3_2D": "H + H₂ 二维反应散射 (Eckart 垒 / LEPS + 含时波包)",
        },''', '''        "methods": ["rhf", "rohf", "uhf", "rks", "roks", "uks", "mp2", "ccsd",
                    "ccsd(t)", "casscf", "nevpt2", "tddft", "eom-ccsd"],
        "solvent_models": ["ddcosmo", "pcm", "ddpcm", "smd"],
        "features": ["几何优化", "内坐标/松弛扫描", "谐振频率", "CI-NEB 过渡态",
                     "IRC 反应路径", "激发态势能面", "自旋-轨道耦合 (单电子 BP)",
                     "隐式溶剂", "数据集/主动学习", "NN 拟合 (含委员会)"],
        "scope": "势能面 (PES) 从头算计算与机器学习拟合",''')
cli.write_text(rw(t), encoding="utf-8")

# ---- 3) core: 只保留 PES 相关 (去 DynamicsResult 与传播诊断) ----
bp = root / pkg / "core" / "base.py"
t = bp.read_text(encoding="utf-8")
if "class DynamicsResult:" in t:
    t = t[: t.index("class DynamicsResult:")].rstrip() + "\n"
    bp.write_text(t, encoding="utf-8")

(root / pkg / "core" / "__init__.py").write_text(
    '"""核心基础 (PES 子集): 量子体系/势能面容器、周期表常量、验证与可复现性工件。"""\n'
    "from .base import QuantumSystem, PES\n"
    "from .validation import (\n"
    "    ValidationError, validate_grid, validate_energy_window,\n"
    "    validate_pes_values, validate_positive, data_fingerprint,\n"
    "    environment_info, write_run_manifest,\n"
    ")\n\n"
    "__all__ = [\n"
    '    "QuantumSystem", "PES",\n'
    '    "ValidationError", "validate_grid", "validate_energy_window",\n'
    '    "validate_pes_values", "validate_positive", "data_fingerprint",\n'
    '    "environment_info", "write_run_manifest",\n]\n', encoding="utf-8")

vp = root / pkg / "core" / "validation.py"
t = vp.read_text(encoding="utf-8")
i0 = t.find("# " + "-" * 75 + "\n# 传播健康诊断")
i1 = t.find("# " + "-" * 75 + "\n# 可复现性")
if i0 > 0 and i1 > i0:
    t = t[:i0] + t[i1:]
t = t.replace("from dataclasses import dataclass, field\n", "")
_old_doc = (
    '"""输入验证与数值健康诊断 — 科学计算软件的质量底线。\n\n'
    "三类检查:\n"
    "1. **输入验证** (validate_*): 网格/能量窗/势能值/参数的边界与有限性,\n"
    "   失败时抛出带可操作提示的 ``ValidationError``;\n"
    "2. **传播健康** (PropagationHealth): 波包传播后的守恒、吸收、能量漂移\n"
    "   诊断 — 这些是量子动力学结果可信度的第一道闸门;\n"
    "3. **步长建议** (suggest_dt): 基于势能幅度与 Nyquist 动能的启发式\n"
    "   相位精度上限 (split-operator 无 CFL 限制, 但 dt 过大时相位误差\n"
    "   与 CAP 吸收效率都会恶化)。\n\n"
    "设计原则: 验证失败立即报错 (fail fast), 健康检查只告警不中断 —\n"
    "物理上可疑但可继续的情形由调用方决定。\n"
    '"""')
_new_doc = (
    '"""输入验证与可复现性工件 — 科学计算软件的质量底线。\n\n'
    "两类检查:\n"
    "1. **输入验证** (``validate_*``): 网格/能量窗/势能值/参数的边界与有限性,\n"
    "   失败时抛出带可操作提示的 ``ValidationError``;\n"
    "2. **可复现性** (``data_fingerprint`` / ``environment_info`` /\n"
    "   ``write_run_manifest``): 数据集指纹与运行环境清单, 保证势能面数据与\n"
    "   拟合结果可溯源。\n\n"
    "设计原则: 验证失败立即报错 (fail fast)。\n"
    '"""')
t = t.replace(_old_doc, _new_doc)
vp.write_text(t, encoding="utf-8")

# ---- 4) nn: 去掉二维传播子桥接函数 ----
mp = root / pkg / "nn" / "model.py"
t = mp.read_text(encoding="utf-8")
if "def nn_pes_2d(" in t:
    i0 = t.index("def nn_pes_2d(")
    m2 = re.search(r"\n(def |class )", t[i0 + 1:])
    t = t[:i0].rstrip() + "\n" + t[i0 + 1 + (m2.start() if m2 else len(t)):]
    mp.write_text(t, encoding="utf-8")
ip = root / pkg / "nn" / "__init__.py"
ip.write_text(ip.read_text(encoding="utf-8").replace(
    "from .model import FeedForwardNN, PESNN, nn_pes_2d",
    "from .model import FeedForwardNN, PESNN"), encoding="utf-8")

# ---- 5) 零散措辞 (单位约定 / 包说明) ----
cp = root / pkg / "pes" / "calculators.py"
t = cp.read_text(encoding="utf-8")
t = t.replace("""单位约定: 坐标 Bohr, 能量 Hartree, 梯度 Hartree/Bohr —— 与动力学
模块一致, 无需换算。""",
              """单位约定: 坐标 Bohr, 能量 Hartree, 梯度 Hartree/Bohr (全程原子单位,
不引入 eV/Å 等换算层, 避免数据集与拟合面之间出现单位漂移)。""")
cp.write_text(t, encoding="utf-8")

ap = root / pkg / "__init__.py"
t = ap.read_text(encoding="utf-8")
t = t.replace("从 AutoQuantumDynamics 单体仓库拆分的**独立子集**:\n",
              "独立的势能面 (PES) 计算与拟合工具包:\n")
ap.write_text(t, encoding="utf-8")

print("✓ 同步与裁剪完成")
PYEOF

echo "✓ 已从上游同步 PES 子集 (README/VALIDATION/pyproject/tools/validation 未被覆盖)"
