import argparse
import sys
import json

from aqpes import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="aqpes",
        description="AutoQuantumPES: 势能面计算与拟合 (PES 子集)",
    )
    parser.add_argument(
        "-V", "--version",
        action="version",
        version=f"AutoQuantum v{__version__}",
    )

    sub = parser.add_subparsers(dest="command", help="可用命令")

    info_parser = sub.add_parser("info", help="显示系统信息")
    info_parser.add_argument(
        "system", nargs="?", default="H2_1D",
        help="体系名称",
    )

    sample_parser = sub.add_parser(
        "sample", help="用电子结构后端生成 NN 训练数据 (npz)")
    sample_parser.add_argument(
        "--backend", default="demo",
        choices=["demo", "analytic", "xtb", "pyscf", "ase"],
        help="电子结构后端 (demo=内置 LJ 演示, 非量子化学)",
    )
    sample_parser.add_argument(
        "--input", required=True, help="参考几何 XYZ 文件 (Bohr)")
    sample_parser.add_argument(
        "-o", "--output", default="data.npz", help="输出数据集 (.npz)")
    sample_parser.add_argument(
        "--n-per-dim", type=int, default=5, help="每维采样点数")
    sample_parser.add_argument(
        "--range", type=float, nargs=2, default=[-0.4, 0.4],
        help="笛卡尔位移范围 (Bohr)")
    sample_parser.add_argument(
        "--active-atoms", type=int, nargs="+", default=None,
        help="参与位移的原子序号 (默认全部)")
    sample_parser.add_argument(
        "--axes", type=int, nargs="+", default=[0, 1, 2],
        help="参与位移的坐标轴 (0/1/2)")
    sample_parser.add_argument(
        "--min-distance", type=float, default=1.2,
        help="最小原子间距过滤 (Bohr)")
    sample_parser.add_argument(
        "--max-points", type=int, default=5000, help="采样点上限")
    sample_parser.add_argument("--charge", type=int, default=0,
                               help="总电荷 (xtb/pyscf)")
    sample_parser.add_argument("--uhf", type=int, default=0,
                               help="未成对电子数 (xtb)")
    sample_parser.add_argument("--spin", type=int, default=None,
                               help="自旋参数 2S = Na - Nb (pyscf, 缺省与 --uhf 保持一致)")
    sample_parser.add_argument("--method", default="rhf",
                               help="电子结构方法: rhf, uhf, rohf, dft, rks, roks, uks (pyscf)")
    sample_parser.add_argument("--basis", default="sto-3g",
                               help="基组 (pyscf, 默认 sto-3g)")
    sample_parser.add_argument("--xc", default=None,
                               help="DFT 泛函名称 (如 b3lyp, pbe; pyscf)")
    sample_parser.add_argument("--spin-lock", action="store_true",
                               help="启用严格自旋态检查与自旋污染截断 (pyscf)")
    sample_parser.add_argument("--spin-tol", type=float, default=0.1,
                               help="自旋锁定允许的最大偏差 |<S^2> - S(S+1)| (默认 0.1)")
    sample_parser.add_argument("--mom", action="store_true",
                               help="启用最大重叠法 (MOM) 沿采样序列跟踪特定激发/占据态 (pyscf)")
    _add_solvent_args(sample_parser)
    _add_casscf_args(sample_parser)

    soc_parser = sub.add_parser(
        "soc", help="自旋-轨道耦合 (单电子 Breit-Pauli): ζ / 精细结构 / 单-三态耦合")
    soc_parser.add_argument("--input", required=True, help="几何 XYZ 文件 (Bohr)")
    soc_parser.add_argument("--basis", default="cc-pvdz")
    soc_parser.add_argument("--charge", type=int, default=0)
    soc_parser.add_argument("--spin", type=int, default=0, help="2S = Na - Nb")
    soc_parser.add_argument("--method", default="rohf",
                            help="SCF 方法 (rhf/rohf/uhf/rks/roks/uks)")
    soc_parser.add_argument("--xc", default=None, help="DFT 泛函")
    soc_parser.add_argument("--orbitals", type=int, nargs="+", default=None,
                            help="p/π 壳层轨道索引 (默认自动识别)")
    soc_parser.add_argument("--term", default="auto", choices=["auto", "P", "Pi"],
                            help="项类型: 原子 ²P (分裂 3ζ/2) 或 ²Π (分裂 |ζ|)")
    soc_parser.add_argument("--z-eff", nargs="+", default=None,
                            help="有效核电荷 元素=Z (经验屏蔽修正), 如 F=5.55")
    soc_parser.add_argument("--active-orbitals", type=int, nargs="+", default=None,
                            help="态相互作用模式: 活性空间轨道索引")
    soc_parser.add_argument("--singlet-roots", type=int, default=1)
    soc_parser.add_argument("--triplet-roots", type=int, default=1)
    soc_parser.add_argument("--nelecas", type=int, default=None,
                            help="活性空间电子数 (默认由占据推断)")

    fit_parser = sub.add_parser(
        "fit", help="在数据集 (npz) 上训练 NN 势能代理面")
    fit_parser.add_argument("--data", required=True, help="数据集 (.npz)")
    fit_parser.add_argument("-o", "--output", default="model.pkl",
                            help="输出模型 (.pkl)")
    fit_parser.add_argument("--epochs", type=int, default=800)
    fit_parser.add_argument("--lr", type=float, default=0.005)
    fit_parser.add_argument("--layers", type=int, nargs="+",
                            default=[64, 64, 32])
    fit_parser.add_argument("--force-weight", type=float, default=1.0,
                            help="力训练权重 (0=纯能量拟合)")
    fit_parser.add_argument("--max-train-points", type=int, default=6000)
    fit_parser.add_argument(
        "--symmetry", action="store_true",
        help="对称函数+共享原子能量模式: 数据集需含 (n,3N) points 与 symbols, "
             "输出置换严格不变的委员会势能面")
    fit_parser.add_argument("--committee", type=int, default=4,
                            help="委员会成员数 (--symmetry, 默认 4)")
    fit_parser.add_argument("--no-gif", action="store_true",
                            help=argparse.SUPPRESS)

    sub.add_parser("backends", help="列出电子结构后端可用性")

    # ---- opt: 几何优化 (极小点) ----
    opt_parser = sub.add_parser("opt", help="几何优化 (BFGS, 解析梯度)")
    _add_calc_args(opt_parser)
    opt_parser.add_argument("-o", "--output", default="optimized.npz",
                            help="输出 npz (含优化几何/能量/梯度)")
    opt_parser.add_argument("--gtol", type=float, default=1e-5,
                            help="梯度收敛阈值 (Hartree/Bohr)")
    opt_parser.add_argument("--max-iter", type=int, default=200)

    # ---- scan: 内坐标 PES 扫描 ----
    sc_parser = sub.add_parser("scan", help="内坐标扫描 -> PES 训练集 (npz)")
    _add_calc_args(sc_parser)
    sc_parser.add_argument("--mode", default="bond",
                           choices=["bond", "angle", "path", "relax-bond"],
                           help="扫描类型")
    sc_parser.add_argument("--atoms", type=int, nargs="+", default=None,
                           help="原子索引: bond i j | angle i j k (顶点 j)")
    sc_parser.add_argument("--range", type=float, nargs=2, default=None,
                           help="扫描范围: 键长 (Bohr) 或键角 (度)")
    sc_parser.add_argument("--n", type=int, default=11, help="扫描点数")
    sc_parser.add_argument("--second", default=None,
                           help="path 模式的第二几何 (XYZ, Bohr)")
    sc_parser.add_argument("-o", "--output", default="scan.npz")

    # ---- freq: 谐振频率 ----
    fr_parser = sub.add_parser("freq", help="谐振频率 (数值 Hessian)")
    _add_calc_args(fr_parser)
    fr_parser.add_argument("--opt-first", action="store_true",
                           help="先做几何优化再算频率")

    args = parser.parse_args()

    if args.command == "info":
        return _show_info(args)
    elif args.command == "sample":
        return _sample_data(args)
    elif args.command == "fit":
        return _fit_nn(args)
    elif args.command == "backends":
        return _show_backends()
    elif args.command == "soc":
        return _cmd_soc(args)
    elif args.command == "opt":
        return _cmd_opt(args)
    elif args.command == "scan":
        return _cmd_scan(args)
    elif args.command == "freq":
        return _cmd_freq(args)
    else:
        parser.print_help()


def _add_casscf_args(p):
    """CASSCF/态平均参数 (sample/opt/scan/freq 共用; pyscf 后端)。"""
    p.add_argument("--active-space", type=int, nargs=2, default=None,
                   metavar=("NCAS", "NELECAS"),
                   help="CASSCF 活性空间 (轨道数 电子数), 如 --active-space 4 4")
    p.add_argument("--pt2", default=None, choices=["nevpt2"],
                   help="CASSCF 之上的动态相关 (NEVPT2; PySCF 无 CASPT2)")
    p.add_argument("--nstates", type=int, default=None,
                   help="态平均态数 (>1 启用态平均 CASSCF 激发态势能面)")
    p.add_argument("--state", type=int, default=None,
                   help="选中的态 (0-based; 需配合 --nstates>1)")
    p.add_argument("--state-weights", type=float, nargs="+", default=None,
                   help="态平均权重 (缺省等权; 长度须等于 --nstates)")
    p.add_argument("--state-average", action="store_true",
                   help="启用态平均 CASSCF (需配合 --nstates>1; 自旋纯)")
    p.add_argument("--follow", action="store_true",
                   help="根跟踪 (态平均 CASSCF: 按上一几何 CI 向量最大重叠选根)")


def _casscf_kwargs(args):
    """从 CLI 参数提取 CASSCF kwargs (未指定时保持默认行为)。"""
    out = {}
    if getattr(args, "active_space", None) is not None:
        out["active_space"] = tuple(args.active_space)
    if getattr(args, "pt2", None):
        out["pt2"] = args.pt2
    if getattr(args, "state_average", False):
        out["state_average"] = True
    if getattr(args, "nstates", None):
        out["nstates"] = args.nstates
    if getattr(args, "state", None) is not None:
        out["state"] = args.state
    if getattr(args, "state_weights", None):
        out["state_weights"] = tuple(args.state_weights)
    if getattr(args, "follow", False):
        out["follow"] = True
    return out


def _add_solvent_args(p):
    """隐式溶剂参数 (sample/opt/scan/freq 共用; pyscf 后端)。"""
    p.add_argument("--solvent", default=None,
                   help="隐式溶剂名 (如 water/methanol; smd 必须用溶剂名)")
    p.add_argument("--solvent-eps", type=float, default=None,
                   help="溶剂介电常数 (直接指定; smd 不支持)")
    p.add_argument("--solvent-model", default="ddcosmo",
                   choices=["ddcosmo", "pcm", "ddpcm", "smd"],
                   help="隐式溶剂模型 (默认 ddcosmo; smd 仅 SCF 层)")
    p.add_argument("--pcm-variant", default="IEF-PCM",
                   choices=["C-PCM", "IEF-PCM", "COSMO", "SS(V)PE"],
                   help="PCM 变体 (仅 --solvent-model pcm)")


def _solvent_kwargs(args):
    """从 CLI 参数提取溶剂 kwargs (未指定溶剂时不传, 保持默认行为)。"""
    if getattr(args, "solvent", None) is None \
            and getattr(args, "solvent_eps", None) is None:
        return {}
    out = {"solvent_model": args.solvent_model,
           "pcm_variant": args.pcm_variant}
    if args.solvent is not None:
        out["solvent"] = args.solvent
    if args.solvent_eps is not None:
        out["solvent_eps"] = args.solvent_eps
    return out


def _add_calc_args(p):
    """opt/scan/freq 共用的后端参数。"""
    p.add_argument("--backend", default="pyscf", choices=["pyscf", "xtb", "demo"])
    p.add_argument("--input", required=True, help="几何 XYZ 文件 (Bohr)")
    p.add_argument("--method", default="rhf",
                   help="rhf/rohf/uhf/dft/rks/roks/uks/mp2/ccsd/ccsd(t)")
    p.add_argument("--basis", default="cc-pvdz")
    p.add_argument("--charge", type=int, default=0)
    p.add_argument("--spin", type=int, default=0, help="2S = Na - Nb")
    p.add_argument("--xc", default=None, help="DFT 泛函")
    p.add_argument("--frozen-core", action="store_true", help="冻结核 (MP2/CCSD)")
    _add_solvent_args(p)
    _add_casscf_args(p)


def _make_calc(args):
    from aqpes.pes.calculators import make_calculator
    symbols, coords = _read_xyz(args.input)
    if args.backend == "pyscf":
        return symbols, coords, make_calculator(
            "pyscf", symbols=symbols, basis=args.basis, charge=args.charge,
            spin=args.spin, method=args.method, xc=args.xc,
            frozen_core=args.frozen_core, **_solvent_kwargs(args),
            **_casscf_kwargs(args))
    return symbols, coords, make_calculator(args.backend, symbols=symbols)


def _cmd_opt(args):
    import numpy as np
    from aqpes.pes.optimize import optimize_geometry
    symbols, coords, calc = _make_calc(args)
    print(f"体系: {len(symbols)} 原子 ({' '.join(symbols)}); "
          f"{args.method}/{args.basis if args.backend == 'pyscf' else args.backend}")
    opt, info = optimize_geometry(calc, np.asarray(coords), gtol=args.gtol,
                                  max_iter=args.max_iter, verbose=True)
    e, g = calc.energy_and_gradient(opt)
    print(f"收敛: {info['converged']} | 迭代 {info['n_iter']} | "
          f"E = {e:.8f} Ha | |g|max = {np.abs(g).max():.2e}")
    print(f"优化几何 (Bohr):\n{np.array2string(opt, precision=6)}")
    np.savez(args.output, coords=opt, energy=e, gradient=g, symbols=symbols,
             converged=info["converged"], method=args.method, basis=args.basis)
    print(f"已保存 → {args.output}")
    return 0


def _cmd_scan(args):
    import numpy as np
    from aqpes.pes import scan as Scan
    symbols, coords, calc = _make_calc(args)
    c = np.asarray(coords)
    if args.mode in ("bond", "relax-bond"):
        i, j = (args.atoms or [0, 1])[:2]
        lo, hi = args.range or (0.8, 2.5)
        grid = np.linspace(lo, hi, args.n)
        fn = (Scan.relaxed_scan_bond if args.mode == "relax-bond"
              else Scan.scan_bond)
        data = fn(calc, symbols, c, i, j, grid)
    elif args.mode == "angle":
        i, j, k = (args.atoms or [0, 1, 2])[:3]
        lo, hi = args.range or (60.0, 180.0)
        data = Scan.scan_angle(calc, symbols, c, i, j, k,
                               np.linspace(lo, hi, args.n))
    else:                                    # path
        if not args.second:
            raise SystemExit("path 模式需要 --second <XYZ>")
        _, c2 = _read_xyz(args.second)
        data = Scan.scan_path(calc, symbols, c, np.asarray(c2), args.n)
    data.save_npz(args.output, provenance=calc.provenance)
    print(f"扫描 {data.points.shape[0]} 点 → {args.output}; "
          f"能量范围 [{data.energies.min():.6f}, {data.energies.max():.6f}] Ha")
    return 0


def _cmd_freq(args):
    import numpy as np
    from aqpes.pes.optimize import (optimize_geometry,
                                          harmonic_frequencies)
    symbols, coords, calc = _make_calc(args)
    c = np.asarray(coords)
    if args.opt_first:
        c, info = optimize_geometry(calc, c)
        print(f"先优化: E = {info['energy']:.8f} Ha, |g|max = {info['grad_max']:.2e}")
    freqs, finf = harmonic_frequencies(calc, symbols, c)
    print(f"谐振频率 (cm^-1, {len(freqs)} 个模式):")
    for k, f in enumerate(freqs, 1):
        print(f"  mode {k:2d}: {f:10.2f}{'   (虚频)' if f < 0 else ''}")
    print(f"虚频数 = {finf['n_imag']}")
    return 0



def _read_xyz(path: str):
    """解析简单 XYZ 文件 (坐标按文件原样视为 Bohr)。"""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    n = int(lines[0].split()[0])
    symbols, coords = [], []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return symbols, coords


def _sample_data(args):
    import numpy as np
    from aqpes.pes.calculators import make_calculator
    from aqpes.pes.abinitio import AbInitioData

    symbols, coords = _read_xyz(args.input)
    print(f"参考几何: {len(symbols)} 原子 ({' '.join(symbols)})")

    kwargs = {}
    if args.backend in ("xtb", "pyscf"):
        kwargs = {"charge": args.charge}
        if args.backend == "xtb":
            kwargs["uhf"] = args.uhf
        elif args.backend == "pyscf":
            spin_val = args.spin if args.spin is not None else args.uhf
            kwargs.update({
                "spin": spin_val,
                "method": args.method,
                "basis": args.basis,
                "xc": args.xc,
                "spin_lock": args.spin_lock,
                "spin_tol": args.spin_tol,
                "use_mom": args.mom,
            })
            kwargs.update(_solvent_kwargs(args))
            kwargs.update(_casscf_kwargs(args))
    calc = make_calculator(args.backend, symbols=symbols, **kwargs)
    print(f"后端: {calc.name} {calc.provenance}")

    data = AbInitioData.sample_geometries(
        calc, np.asarray(coords), ranges=(tuple(args.range),),
        n_per_dim=args.n_per_dim, active_atoms=args.active_atoms,
        axes=args.axes, min_distance=args.min_distance,
        max_points=args.max_points, verbose=True)
    data.save_npz(args.output)
    print(f"已保存 {data.n_points} 个构型 → {args.output}")
    print(f"能量范围: [{data.energies.min():.6f}, {data.energies.max():.6f}] Hartree")
    return 0


def _cmd_soc(args):
    """自旋-轨道耦合: 轨道层 ζ/分裂 或 态相互作用 SOC 矩阵。"""
    from aqpes.pes.calculators import make_calculator

    symbols, coords = _read_xyz(args.input)
    z_eff = None
    if args.z_eff:
        z_eff = {}
        for item in args.z_eff:
            if "=" not in item:
                print(f"--z-eff 需要 元素=Z 形式, 得到 {item!r}")
                return 2
            k, v = item.split("=", 1)
            z_eff[k.strip()] = float(v)
    calc = make_calculator("pyscf", symbols=symbols, basis=args.basis,
                           charge=args.charge, spin=args.spin,
                           method=args.method, xc=args.xc)
    print(f"体系: {len(symbols)} 原子 ({' '.join(symbols)}) | "
          f"{calc.provenance['method']}/{calc.provenance['basis']} | "
          f"spin={args.spin}")
    if args.active_orbitals:
        out = calc.soc_state_interaction(
            coords, args.active_orbitals, singlet_roots=args.singlet_roots,
            triplet_roots=args.triplet_roots, z_eff=z_eff, nelecas=args.nelecas)
        e = out["energies_cm"]
        H = out["soc_cm"]
        print(f"活性空间: 轨道 {args.active_orbitals}, nelecas = {out['nelecas']}")
        print("态能量 (cm⁻¹, 相对最低): " + " | ".join(
            f"{lbl} {ev:.1f}" for lbl, ev in zip(out["labels"], e)))
        print("SOC 矩阵 |H_SO| (cm⁻¹):")
        for i, lbl in enumerate(out["labels"]):
            print("  " + lbl.ljust(12) + " ".join(
                f"{abs(H[i, j]):9.3f}" for j in range(len(out["labels"]))))
        return 0
    out = calc.soc_terms(coords, orbitals=args.orbitals, z_eff=z_eff,
                         term=args.term)
    print(f"p/π 壳层轨道: {out['orbitals']} | 项类型: {out['term']}")
    print(f"ζ = {out['zeta_cm']:.3f} cm⁻¹ → 精细结构分裂 = "
          f"{out['splitting_cm']:.3f} cm⁻¹")
    print("(单电子 Breit–Pauli; 二电子 SOC 未含 → 轻元素系统性偏大, "
          "重元素更接近实验; 可用 --z-eff 做经验屏蔽修正)")
    return 0


def _fit_nn(args):
    import numpy as np
    from aqpes.pes.abinitio import AbInitioData
    from aqpes.nn import NNTrainer, TrainingConfig
    from aqpes.nn.model import PESNN

    data = AbInitioData.load_npz(args.data)
    if args.symmetry:
        return _fit_symmetry_committee(args, data)

    prov = getattr(data, "provenance", None)
    if prov:
        print(f"数据来源: {prov}")
    print(f"训练点: {data.n_points}, 特征维度: {data.points.shape[1]}")

    dY = data.gradients if (args.force_weight > 0
                            and data.gradients is not None) else None
    if dY is not None and dY.shape != data.points.shape:
        dY = np.asarray(dY).reshape(dY.shape[0], -1)  # (n,N,3) → (n,3N)
    if args.force_weight > 0 and dY is None:
        print("  [warn] 数据集无梯度, 退化为纯能量拟合")

    config = TrainingConfig(hidden_layers=args.layers, epochs=args.epochs,
                            lr=args.lr, force_weight=args.force_weight,
                            max_train_points=args.max_train_points)
    model, history = NNTrainer(config).train(data.points, data.energies, dY=dY)

    pred = model.predict(data.points)
    rmse = float(np.sqrt(np.mean((pred - data.energies) ** 2)))
    span = float(data.energies.max() - data.energies.min())
    print(f"能量 RMSE: {rmse:.3e} Hartree ({rmse / span * 100:.2f}% of span)")
    if data.gradients is not None:
        g_ref = np.asarray(data.gradients).reshape(data.gradients.shape[0], -1)
        g_rmse = float(np.sqrt(np.mean(
            (model.gradient(data.points) - g_ref) ** 2)))
        print(f"梯度 RMSE: {g_rmse:.3e} Hartree/Bohr")

    model.save(args.output)
    print(f"模型已保存 → {args.output}")
    print("用法: from aqpes.nn.model import PESNN; "
          "model = PESNN.load(path); model.predict(points)")
    return 0


def _fit_symmetry_committee(args, data):
    """对称函数 + 共享原子能量委员会 (置换/平移/旋转严格不变)。"""
    import numpy as np
    from aqpes.nn.ensemble import (train_atomic_committee,
                                         AtomicTrainingConfig)
    from aqpes.nn.symmetry import SymmetryFunctionParams

    if data.points.ndim != 2 or data.points.shape[1] % 3 != 0:
        raise SystemExit("--symmetry 需要 (n, 3N) 笛卡尔坐标数据集")
    symbols = list(data.symbols) if getattr(data, "symbols", None) else None
    if not symbols or len(symbols) * 3 != data.points.shape[1]:
        raise SystemExit("--symmetry 需要数据集内含 symbols (元素序列, 长度 N)")
    coords = data.points.reshape(-1, len(symbols), 3)
    print(f"对称函数模式: {len(symbols)} 原子 {symbols}, "
          f"{coords.shape[0]} 构型, 委员会 {args.committee} 成员")
    committee, info = train_atomic_committee(
        symbols, coords, data.energies, n_models=args.committee,
        config=AtomicTrainingConfig(epochs=args.epochs),
        seed=0)
    print(f"总能量 RMSE: {info['rmse']:.3e} Hartree "
          f"(特征维度 {info['n_features']}/中心)")
    committee.save(args.output)
    print(f"委员会已保存 → {args.output} (置换不变, OOD 不确定性: committee.std(coords))")
    return 0
    prov = getattr(data, "provenance", None)
    if prov:
        print(f"数据来源: {prov}")
    print(f"训练点: {data.n_points}, 特征维度: {data.points.shape[1]}")

    dY = data.gradients if (args.force_weight > 0
                            and data.gradients is not None) else None
    if dY is not None and dY.shape != data.points.shape:
        dY = np.asarray(dY).reshape(dY.shape[0], -1)  # (n,N,3) → (n,3N)
    if dY is not None:
        # 几何梯度 (n, N_atoms, 3) → 与展平笛卡尔特征 (n, 3N) 对齐
        dY = np.asarray(dY).reshape(dY.shape[0], -1)
    if args.force_weight > 0 and dY is None:
        print("  [warn] 数据集无梯度, 退化为纯能量拟合")

    config = TrainingConfig(hidden_layers=args.layers, epochs=args.epochs,
                            lr=args.lr, force_weight=args.force_weight,
                            max_train_points=args.max_train_points)
    model, history = NNTrainer(config).train(data.points, data.energies, dY=dY)

    pred = model.predict(data.points)
    rmse = float(np.sqrt(np.mean((pred - data.energies) ** 2)))
    span = float(data.energies.max() - data.energies.min())
    print(f"能量 RMSE: {rmse:.3e} Hartree ({rmse / span * 100:.2f}% of span)")
    if data.gradients is not None:
        g_ref = np.asarray(data.gradients).reshape(data.gradients.shape[0], -1)
        g_rmse = float(np.sqrt(np.mean(
            (model.gradient(data.points) - g_ref) ** 2)))
        print(f"梯度 RMSE: {g_rmse:.3e} Hartree/Bohr")

    model.save(args.output)
    print(f"模型已保存 → {args.output}")
    print("用法: from aqpes.nn.model import PESNN; "
          "model = PESNN.load(path); model.predict(points)")
    return 0


def _show_backends():
    import json
    from aqpes.pes.calculators import available_calculators
    print(json.dumps(available_calculators(), indent=2, ensure_ascii=False))
    return 0


def _show_info(args):
    info = {
        "version": __version__,
        "system": args.system,
        "available_pes": ["morse", "harmonic", "lj", "leps", "eckart"],
        "methods": ["rhf", "rohf", "uhf", "rks", "roks", "uks", "mp2", "ccsd",
                    "ccsd(t)", "casscf", "nevpt2", "tddft", "eom-ccsd"],
        "solvent_models": ["ddcosmo", "pcm", "ddpcm", "smd"],
        "features": ["几何优化", "内坐标/松弛扫描", "谐振频率", "CI-NEB 过渡态",
                     "IRC 反应路径", "激发态势能面", "自旋-轨道耦合 (单电子 BP)",
                     "隐式溶剂", "数据集/主动学习", "NN 拟合 (含委员会)"],
        "scope": "势能面 (PES) 从头算计算与机器学习拟合",
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
