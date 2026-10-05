#!/usr/bin/env bash
# 从主仓库 (AutoQuantumDynamics) 重新生成本独立包的**可复现**同步脚本。
#
# 用法:  bash tools/sync_from_monorepo.sh <主仓库路径>       # 默认 ..
# 说明:  只同步势能面相关子集 (pes/ + nn/ + core 子集 + CLI), 并改写 import
#        前缀 autoquantum. → aqpes.; core/__init__ 与 cli 会做裁剪 (去掉
#        依赖波包动力学/可视化的 engine 与 run 命令); 本脚本 **不覆盖**
#        README.md / VALIDATION.md / pyproject.toml / tools/ / validation/。
set -euo pipefail
SRC="${1:-..}"
DST="$(cd "$(dirname "$0")/.." && pwd)"
PKG=aqpes
[ -d "$SRC/autoquantum/pes" ] || { echo "找不到 $SRC/autoquantum/pes"; exit 1; }

rm -rf "$DST/$PKG/pes" "$DST/$PKG/nn" "$DST/$PKG/cli.py"
mkdir -p "$DST/$PKG/pes" "$DST/$PKG/nn" "$DST/$PKG/core" "$DST/tests"
cp "$SRC"/autoquantum/pes/*.py "$DST/$PKG/pes/"
cp "$SRC"/autoquantum/nn/*.py  "$DST/$PKG/nn/"
for f in base.py periodic.py validation.py; do cp "$SRC/autoquantum/core/$f" "$DST/$PKG/core/"; done
cp "$SRC/autoquantum/cli.py" "$DST/$PKG/cli.py"

python3 - "$DST" "$PKG" << 'PY'
import re, sys, pathlib
dst, pkg = sys.argv[1], sys.argv[2]
root = pathlib.Path(dst)
def rw(t):
    t = t.replace("from autoquantum.", f"from {pkg}.")
    t = t.replace("import autoquantum.", f"import {pkg}.")
    t = t.replace("from autoquantum import", f"from {pkg} import")
    return t
for p in list(root.glob(f"{pkg}/**/*.py")):
    p.write_text(rw(p.read_text(encoding="utf-8")), encoding="utf-8")
cli = root / pkg / "cli.py"
t = cli.read_text(encoding="utf-8")
t = t.replace("from autoquantum.core.engine import AutoPipeline, PipelineConfig\n", "")
t = re.sub(r"    run_parser = sub\.add_parser\(.*?\n\n    info_parser", "    info_parser", t, flags=re.S)
t = t.replace('    if args.command == "run":\n        return _run_pipeline(args)\n    elif args.command == "info":', '    if args.command == "info":')
t = re.sub(r"\ndef _run_pipeline\(args\):.*?\n\ndef _read_xyz", "\n\ndef _read_xyz", t, flags=re.S)
t = t.replace("AutoQuantum: 分子反应动力学", "AutoQuantumPES: 势能面计算与拟合")
cli.write_text(rw(t), encoding="utf-8")
# 测试
for name in ("test_calculators","test_pes","test_leps","test_abinitio",
             "test_optimize_scan","test_nn","test_symmetry","test_soc",
             "test_data_pipeline"):
    src = pathlib.Path(sys.argv[1]).parent / "tests" / f"{name}.py"
    if src.exists():
        (root / "tests" / f"{name}.py").write_text(
            rw(src.read_text(encoding="utf-8")), encoding="utf-8")
PY
echo "✓ 已从 $SRC 同步 (core/__init__.py, __init__.py 请勿覆盖: 由本包维护)"
