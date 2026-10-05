"""自旋-轨道耦合 (SOC) — 单电子 Breit–Pauli 近似。

    H_SO = (α²/2) Σ_i Σ_A Z_A^eff (l_iA · s_i) / r_iA³

三层接口
--------
1. **积分层**: :func:`soc_integrals` / :func:`soc_integrals_mo`
   逐原子构造 (l 与 r³ 均相对该原子, 用 ``with_rinv_at_nucleus``), 含 α²/2
   前置因子, 单位为 Hartree。结果与分子在空间中的位置无关 (验证脚本检验)。
2. **轨道层**: :func:`soc_zeta` → 轨道型 SOC 常数 ζ (cm⁻¹)。
   在 p/π 子空间内对 h_z 求本征值, ζ = max|λ| (与轨道相位/任意混合无关)。
   精细结构分裂 (由 L·S 耦合推出, 见 `splitting_atomic_p`/`splitting_pi`):
   - 原子 ²P (p¹ 或 p⁵): Δ = (3/2)·ζ  [精确类氢检验: ζ = α²Z⁴/48 ⇒ Δ = α²Z⁴/32]
   - 线性分子 ²Π:      Δ = |ζ|      (A 常数; H_SO = A·L_z·S_z)
3. **态相互作用层**: :func:`soc_matrix_states` → 用 CI 矢量 (CASCI/CASSCF) 与
   自旋翻转跃迁密度计算多态 SOC 矩阵 (cm⁻¹), 覆盖单重态-三重态耦合;
   :func:`so_coupled_energies` 给出精细结构分辨的耦合势 (SOC-PES)。

诚实边界
--------
- 仅**单电子** Breit–Pauli 项: 二电子 SOC (含自旋-同自旋与芯层屏蔽修正) 未实现
  → 对轻元素系统性偏离实验, 验证脚本给出**逐元素实测比值**;
- 提供 ``z_eff`` 逐元素经验缩放 (模拟芯层屏蔽), 使用时须明确标注为经验修正;
- state-interaction 要求**共同轨道基** (同一分子的不同自旋流行共享轨道),
  未实现混合自旋的 state_average_mix_ 轨道优化。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# CODATA 精细结构常数与单位换算
ALPHA = 1.0 / 137.035999084
AU2CM = 219474.6313632
#: 单电子 BP 前置因子 α²/2
PREF = ALPHA ** 2 / 2.0

__all__ = [
    "ALPHA", "AU2CM", "PREF",
    "axis_angular_momentum",
    "soc_integrals", "soc_integrals_mo", "soc_zeta",
    "splitting_atomic_p", "splitting_pi",
    "trans_density_1body", "soc_matrix_states",
    "so_coupled_energies", "auto_soc_orbitals",
    "spin_pure_solver", "casci_ci_vectors",
    "soc_orbital_analysis", "infer_nelecas", "soc_state_interaction",
]


# ---------------------------------------------------------------------------
# 1. 积分层
# ---------------------------------------------------------------------------
def soc_integrals(mol, z_eff: Optional[Dict[str, float]] = None) -> np.ndarray:
    """AO 基下三分量单电子 BP SOC 积分 (Hartree, 已含 α²/2)。

    参数
    ----
    mol : pyscf.gto.Mole
    z_eff : {元素符号: 有效核电荷}; None = 用裸核电荷 Z。

    返回
    ----
    ``(3, nao, nao)`` 数组, 三个分量对应 ``l_x, l_y, l_z``。
    """
    nao = mol.nao
    h = np.zeros((3, nao, nao))
    for ia in range(mol.natm):
        sym = mol.atom_symbol(ia)
        z = float(z_eff[sym]) if (z_eff and sym in z_eff) else mol.atom_charge(ia)
        if z == 0.0:
            continue
        # ⚠ 起点约定 (实测踩坑): int1e_prinvxp 的 1/r³ 起点由 **rinv 原点** 控制;
        # with_common_origin 对该积分**无效** —— 平移分子后结果会漂移
        # (实测: He⁺ 从原点移到 (3.1,−2.4,5.7) Bohr, ζ 由 3.874 变成 0.005 cm⁻¹)。
        # 正确做法是把 rinv 原点放到该原子核上。
        try:
            ctx = mol.with_rinv_at_nucleus(ia)
        except AttributeError:                      # 兼容旧版 PySCF
            ctx = mol.with_rinv_origin(mol.atom_coord(ia))
        with ctx:
            h += z * np.asarray(mol.intor("int1e_prinvxp", comp=3), dtype=float)
    # ⚠ 关键约定: libcint 返回 (r×∇) 的**实**矩阵, 而物理角动量 l = −i(r×∇)
    # → 必须乘 −1j 才得到 Hermitian 的 SOC 单电子算符。
    # 漏掉该因子会使矩阵变成反 Hermitian (实反对称), 态相互作用层的矩阵元
    # 在 Hermitian 对称化时被**完全抵消为零** (实测踩过的坑)。
    # 轨道层的 ζ 只取本征值模, 对该相位因子不敏感, 故此前未暴露。
    return (-1j * PREF) * h


def soc_integrals_mo(mol, mo_coeff, z_eff: Optional[Dict[str, float]] = None
                     ) -> np.ndarray:
    """MO 基 SOC 积分 ``(3, nmo, nmo)`` (Hartree)。仅支持单一轨道基 (RHF/ROHF/CASSCF)。"""
    mo = np.asarray(mo_coeff)
    if mo.ndim != 2:
        raise ValueError("SOC 目前仅支持统一轨道基 (RHF/ROHF/CASSCF), "
                         "UHF/UKS 的 alpha/beta 分离轨道不受支持")
    h = soc_integrals(mol, z_eff)
    return np.einsum("kpq,pi,qj->kij", h, mo, mo)


# ---------------------------------------------------------------------------
# 2. 轨道层
# ---------------------------------------------------------------------------
def soc_zeta(h_mo: np.ndarray, orbitals: Sequence[int]) -> float:
    """轨道子空间 (p/π 壳层) 的 SOC 常数 ζ (cm⁻¹), 与相位/混合无关。

    做法: h_z 在实轨道基下为纯虚本征值 (0, ±iζ); 对 ``i·h_z`` 做 Hermitian
    对角化取 max|λ|。子空间应包含完整 p (3 个) 或 π (2 个) 简并组。
    """
    idx = list(orbitals)
    blk = np.asarray(h_mo)[2][np.ix_(idx, idx)]
    # h_z 在实轨道基下为纯虚反对称 (Hermitian 算符) → 本征值 (0, ±iζ);
    # 取 |本征值| 最大值即 ζ, 且对"实反对称"表示 (未乘 −i) 同样成立。
    ev = np.linalg.eigvals(blk)
    return float(np.abs(ev).max() * AU2CM)


def splitting_atomic_p(zeta_cm: float) -> float:
    """原子 ²P 精细结构分裂 Δ(²P₃/₂ − ²P₁/₂) = (3/2)ζ (cm⁻¹)。"""
    return 1.5 * abs(zeta_cm)


def splitting_pi(zeta_cm: float) -> float:
    """线性分子 ²Π 精细结构分裂 |Δ(²Π₃/₂ − ²Π₁/₂)| = |A| = |ζ| (cm⁻¹)。"""
    return abs(zeta_cm)


def _mo_character(mol, mo) -> np.ndarray:
    """MO 的 (原子, l) 布居特征矩阵 (每列归一), 用于同壳层识别。"""
    nao = mol.nao
    ao_atom = np.zeros(nao, dtype=int)
    ao_l = np.zeros(nao, dtype=int)
    idx = 0
    for ib in range(mol.nbas):
        a = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        for _ in range(mol.bas_nctr(ib)):
            ao_atom[idx:idx + 2 * l + 1] = a
            ao_l[idx:idx + 2 * l + 1] = l
            idx += 2 * l + 1
    keys = sorted(set(zip(ao_atom.tolist(), ao_l.tolist())))
    m = np.zeros((len(keys), np.asarray(mo).shape[1]))
    for k, (a, l) in enumerate(keys):
        sel = (ao_atom == a) & (ao_l == l)
        m[k] = (np.asarray(mo)[sel, :] ** 2).sum(axis=0)
    tot = np.maximum(m.sum(axis=0, keepdims=True), 1e-30)
    return m / tot


def _is_linear(mol, tol: float = 1e-3) -> bool:
    """所有原子共线 (容差 Bohr)。"""
    c = np.asarray(mol.atom_coords(), dtype=float)
    if len(c) < 3:
        return True
    v = c[-1] - c[0]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return False
    v = v / n
    rel = c - c[0]
    perp = rel - np.outer(rel @ v, v)
    return bool(np.abs(perp).max() < tol)


def axis_angular_momentum(mol, mo, origin=None) -> np.ndarray:
    """MO 基下沿分子轴 (首-末原子连线) 的角动量矩阵 ``L_axis``。

    用于线性分子的 π/σ 判别: 对 π 轨道 (|Λ|=1) 与 σ 轨道 (Λ=0) 构成的
    2×2 块, ``1j * L_axis`` 的本征值模 = |Λ| (与相位/实轨道混合无关)。
    """
    c = np.asarray(mol.atom_coords(), dtype=float)
    if len(c) < 2:
        raise ValueError("单原子无分子轴, 无法用轴向角动量判别")
    v = c[-1] - c[0]
    v = v / np.linalg.norm(v)
    o = c[0] if origin is None else np.asarray(origin, dtype=float)
    # int1e_rxp / int1e_cg_irxp 用 common origin; 该算符无 1/r 项,
    # 但为稳妥同时在 rinv 原点上也设一次 (对无 1/r 的算符无影响)。
    try:
        cm = mol.with_common_origin(o)
    except AttributeError:
        cm = mol.with_common_orig(o)
    with cm:
        rxp = np.asarray(mol.intor("int1e_cg_irxp", comp=3), dtype=float)
    l_ao = np.einsum("k,kpq->pq", v, rxp)
    mo = np.asarray(mo)
    # 同 soc_integrals: libcint 给实矩阵, 物理 L = −i(r×∇)
    return -1j * (mo.T @ l_ao @ mo)


def _lambda_abs(l_axis: np.ndarray, idx) -> float:
    """子空间角动量本征值模 (π → ≈1, σ → ≈0; 与相位/实虚表示无关)。"""
    idx = list(idx)
    blk = np.asarray(l_axis)[np.ix_(idx, idx)]
    return float(np.abs(np.linalg.eigvals(blk)).max())


def auto_soc_orbitals(mol, mf, tol: float = 1e-4, n_take: int = 3
                      ) -> List[int]:
    """自动选取 p/π 壳层轨道 (原子 ²P 取 3 个; 线性分子 ²Π 取 2 个)。

    规则 (两种情形):
      1. **原子/多重占据**: 若存在 ≥3 个占据 (occ>1e-6) 的 p 型轨道,
         取 mo_energy 最高的 3 个 (价层 p 壳层; ROHF 对简并开壳层的
         ``mo_energy`` 定义可能不同, 故不依赖能量简并判据);
      2. **分子开壳层**: 以最高部分占据轨道为前沿, 用 (原子, l) 布居特征的
         余弦相似度 (>0.99) 找到同壳层伙伴 (可含空轨道), 如 ²Π 的 π 对。
    """
    mo = np.asarray(mf.mo_coeff)
    if mo.ndim != 2:
        raise ValueError("auto_soc_orbitals 仅支持统一轨道基 (RHF/ROHF)")
    occ = np.asarray(mf.mo_occ, dtype=float).ravel()
    e = np.asarray(mf.mo_energy, dtype=float).ravel()
    nao, nmo = mo.shape
    ao_l = np.zeros(nao, dtype=int)
    idx = 0
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        for _ in range(mol.bas_nctr(ib)):
            ao_l[idx:idx + 2 * l + 1] = l
            idx += 2 * l + 1
    pchar = np.array([float(((mo[:, i] ** 2) * ao_l).sum()
                            / max((mo[:, i] ** 2).sum(), 1e-30))
                      for i in range(nmo)])
    part = [i for i in range(nmo) if 0.0 < occ[i] < 2.0]
    # --- 线性分子: 用轴向角动量 |Λ| 判别 π (1) vs σ (0) ---
    # 注: 必须先于 "占据 p 型轨道 ≥3 取最高 3 个" 的原子分支, 否则 OH 这类
    # σ+π 开壳层会错配成 (σ, π) 对 (实测 ζ 变成 σ–π 矩阵元, 假值)。
    if part and mol.natm > 1 and _is_linear(mol):
        frontier = max(part, key=lambda i: e[i])
        try:
            l_axis = axis_angular_momentum(mol, mo)
        except Exception:
            l_axis = None
        if l_axis is not None:
            lam0 = _lambda_abs(l_axis, [frontier])
            if lam0 < 0.3:                       # 前沿为 π 型 → 找同 |Λ| 伙伴
                cand = []
                for j in range(nmo):
                    if j == frontier:
                        continue
                    lam = _lambda_abs(l_axis, [frontier, j])
                    if lam > 0.7:
                        cand.append((abs(e[j] - e[frontier]), j, lam))
                if cand:
                    cand.sort()
                    return sorted([frontier, cand[0][1]])
    ptype_occ = [i for i in range(nmo) if pchar[i] > 0.8 and occ[i] > 1e-6]
    if len(ptype_occ) >= 3:
        return sorted(sorted(ptype_occ, key=lambda i: e[i])[-3:])
    if not part:
        raise ValueError("无部分占据轨道且占据 p 型轨道 <3: 无法自动定位 p/π 壳层; "
                         "请显式给出 orbitals=[...]")
    frontier = max(part, key=lambda i: e[i])
    char = _mo_character(mol, mo)
    ref = char[:, frontier]
    sim = (char * ref[:, None]).sum(axis=0) / np.maximum(
        np.linalg.norm(char, axis=0) * np.linalg.norm(ref), 1e-30)
    cand = [i for i in range(nmo) if sim[i] > 0.99]
    cand = sorted(cand, key=lambda i: (abs(e[i] - e[frontier]), i))
    return sorted(cand[:n_take])


# ---------------------------------------------------------------------------
# 3. 态相互作用层: CI 矢量 + 自旋翻转跃迁密度
# ---------------------------------------------------------------------------
def _strings(norb: int, nele: int):
    """返回 ``(bits, addr)``: 占据位串与它们在 PySCF CI 矢量中的**行/列地址**。

    关键: ``cistring.gen_occslst`` 给出的**顺序**不一定是 PySCF CI 布局的地址
    顺序 (CI 矢量按 ``cistring.strs2addr`` 定址) → 必须显式取地址并按地址排序,
    否则 CI 系数与行列式错配 (实测将使 SOC 矩阵元假零)。
    """
    from pyscf.fci import cistring
    occ = np.asarray(cistring.gen_occslst(range(norb), nele))
    nstr = occ.shape[0]
    bits = np.zeros(nstr, dtype=np.int64)
    if occ.ndim == 2:
        for k in range(occ.shape[1]):
            bits |= (np.int64(1) << occ[:, k].astype(np.int64))
    elif occ.ndim == 1 and occ.size and int(nele) == 1:
        bits = (np.int64(1) << occ.astype(np.int64))
    try:
        addr = np.asarray(cistring.strs2addr(norb, int(nele), bits),
                          dtype=np.int64).ravel()
    except Exception:                      # 极端情形回退 (保持自洽顺序)
        addr = np.arange(nstr, dtype=np.int64)
    order = np.argsort(addr)
    return bits[order], addr[order]


def _parity(mask: np.ndarray, p: int) -> np.ndarray:
    """费米子符号 (-1)^{# 已占据轨道 index < p} (逐字符串, 向量化)。"""
    low = mask & ((1 << p) - 1)
    try:
        cnt = np.bitwise_count(low)                      # numpy ≥ 2.0
    except AttributeError:                                # pragma: no cover
        cnt = np.array([bin(int(x)).count("1") for x in low])
    return np.where(cnt % 2 == 1, -1.0, 1.0)


def _reduced_map(bits: np.ndarray, orb: int):
    """对含 ``orb`` 的位串, 返回 ``(idx, reduced_bits, sign)`` (产生/湮灭共用)。"""
    has = ((bits >> orb) & 1).astype(bool)
    idx = np.nonzero(has)[0]
    sign = _parity(bits[idx], orb)
    return idx, bits[idx] & ~(1 << orb), sign


def _join(key_b, flat_b, sign_b, key_k, flat_k, sign_k):
    """按键求交, 返回对齐的 ``(flat_b, flat_k, sign_b·sign_k)``。"""
    ob = np.argsort(key_b, kind="stable")
    ok = np.argsort(key_k, kind="stable")
    kb, kk = key_b[ob], key_k[ok]
    common, ib, ik = np.intersect1d(kb, kk, assume_unique=False,
                                    return_indices=True)
    if common.size == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=float))
    return (flat_b[ob][ib], flat_k[ok][ik],
            sign_b[ob][ib] * sign_k[ok][ik])


def trans_density_1body(block: str, cibra: np.ndarray, ciket: np.ndarray,
                        norb: int, nelec_ket: Tuple[int, int]) -> np.ndarray:
    """跃迁 1-体密度 ``D[p, q] = <bra| a†_{p,σ} a_{q,σ'} |ket>``。

    ``block``:
      - ``"aa"``: σ=α, σ'=α (bra 扇区 = ket 扇区)
      - ``"bb"``: σ=β, σ'=β (bra 扇区 = ket 扇区)
      - ``"ab"``: σ=α, σ'=β (bra 扇区 = (nα+1, nβ−1))
      - ``"ba"``: σ=β, σ'=α (bra 扇区 = (nα−1, nβ+1))

    索引约定与 ``H = Σ_{pq} h[p,q] a†_{p}a_{q}`` 一致 (p=产生, q=湮灭),
    直接用于 ``Σ_{pq} h[p,q]·D[p,q]``。CI 矢量按 PySCF 布局
    (``cistring.strs2addr``) 定址; 轨道编号 0..norb-1 对应**活性空间**轨道。
    """
    na_k, nb_k = nelec_ket
    sectors = {"aa": (na_k, nb_k), "bb": (na_k, nb_k),
               "ab": (na_k + 1, nb_k - 1), "ba": (na_k - 1, nb_k + 1)}
    if block not in sectors:
        raise ValueError(f"未知 block: {block!r}")
    na_b, nb_b = sectors[block]
    create_spin = "a" if block in ("aa", "ab") else "b"
    annih_spin = "a" if block in ("aa", "ba") else "b"

    bk = {"a": _strings(norb, na_k), "b": _strings(norb, nb_k)}
    bb = {"a": _strings(norb, na_b), "b": _strings(norb, nb_b)}
    ck = np.asarray(ciket).reshape(len(bk["a"][0]), len(bk["b"][0]))
    cb = np.asarray(cibra).reshape(len(bb["a"][0]), len(bb["b"][0]))
    n_ak, n_bk = ck.shape
    n_ab, n_bb = cb.shape
    BIG = 1 << norb
    one = np.int64(1)

    # 位串网格 (用于置换/符号) 与地址网格 (用于 CI 平铺索引)
    bits_b = {"a": bb["a"][0][:, None] * np.ones((1, n_bb), np.int64),
              "b": np.ones((n_ab, 1), np.int64) * bb["b"][0][None, :]}
    bits_k = {"a": bk["a"][0][:, None] * np.ones((1, n_bk), np.int64),
              "b": np.ones((n_ak, 1), np.int64) * bk["b"][0][None, :]}
    flat_b = bb["a"][1][:, None] * n_bb + bb["b"][1][None, :]
    flat_k = bk["a"][1][:, None] * n_bk + bk["b"][1][None, :]

    D = np.zeros((norb, norb))
    for p in range(norb):
        ic, mc, sc = _reduced_map(bb[create_spin][0], p)
        if ic.size == 0:
            continue
        if create_spin == "a":
            bra_key = (mc[:, None] * BIG + bits_b["b"][ic, :]).ravel()
            bra_addr = flat_b[ic, :].ravel()
            bra_sign = np.repeat(sc, n_bb)
        else:
            bra_key = (bits_b["a"][:, ic] * BIG + mc[None, :]).ravel()
            bra_addr = flat_b[:, ic].ravel()
            bra_sign = np.tile(sc, n_ab)
        for q in range(norb):
            ia, ma, sa = _reduced_map(bk[annih_spin][0], q)
            if ia.size == 0:
                continue
            if annih_spin == "a":
                ket_key = (ma[:, None] * BIG + bits_k["b"][ia, :]).ravel()
                ket_addr = flat_k[ia, :].ravel()
                ket_sign = np.repeat(sa, n_bk)
            else:
                ket_key = (bits_k["a"][:, ia] * BIG + ma[None, :]).ravel()
                ket_addr = flat_k[:, ia].ravel()
                ket_sign = np.tile(sa, n_ak)
            fb, fk, ss = _join(bra_key, bra_addr, bra_sign,
                               ket_key, ket_addr, ket_sign)
            if fb.size:
                D[p, q] = float(np.sum(cb.ravel()[fb] * ck.ravel()[fk] * ss))
    return D


def soc_matrix_states(h_mo: np.ndarray, states: List[dict]) -> np.ndarray:
    """多态 SOC 矩阵 (cm⁻¹)。

    参数
    ----
    h_mo : ``(3, ncas, ncas)`` 活性空间 MO 基 SOC 积分 (Hartree)
    states : 每个元素为 ``{"ci": ndarray, "nelec": (na, nb), "label": str}``,
        CI 矢量位于**共同**轨道基与**共同**活性空间。

    返回
    ----
    ``(n, n)`` 复 Hermitian 矩阵 (cm⁻¹)。

    公式 (自旋-轨道算符按 1-体算符展开):
      H_SO = Σ_k Σ_{pq} h_k[p,q]·a†_{pσ}(s_k)_{σσ'}a_{qσ'}
      s_z = (n_α − n_β)/2, s_+ = a†_α a_β, s_− = a†_β a_α  ⇒
      H_SO = Σ_{pq} { h_z[p,q]·(D^{aa}−D^{bb})/2
                    + (h_x−i h_y)[p,q]·D^{ab}/2
                    + (h_x+i h_y)[p,q]·D^{ba}/2 }
    """
    n = len(states)
    out = np.zeros((n, n), dtype=complex)
    hx, hy, hz = (np.asarray(h_mo)[k] for k in range(3))   # 保留复型 (勿转 float)
    hxy_m = 0.5 * (hx - 1j * hy)
    hxy_p = 0.5 * (hx + 1j * hy)
    for i in range(n):
        for j in range(n):
            si, sj = states[i], states[j]
            ci = np.asarray(si["ci"]).ravel()
            cj = np.asarray(sj["ci"]).ravel()
            nik, njk = si["nelec"], sj["nelec"]
            # 归一化 (CASCI 矢量应已归一, 防御性处理)
            ci = ci / np.linalg.norm(ci)
            cj = cj / np.linalg.norm(cj)
            ncas = hz.shape[0]
            val = 0.0 + 0.0j
            # Z 分量 (同扇区)
            if nik == njk:
                daa = trans_density_1body("aa", ci, cj, ncas, nik)
                dbb = trans_density_1body("bb", ci, cj, ncas, nik)
                val += complex(np.sum(hz * (0.5 * (daa - dbb))))
            # 自旋翻转: bra 的 (na,nb) 比 ket 多 (1,−1) → D^{ab}
            if nik == (njk[0] + 1, njk[1] - 1):
                dab = trans_density_1body("ab", ci, cj, ncas, njk)
                val += complex(np.sum(hxy_m * dab))
            if nik == (njk[0] - 1, njk[1] + 1):
                dba = trans_density_1body("ba", ci, cj, ncas, njk)
                val += complex(np.sum(hxy_p * dba))
            out[i, j] = val
    out = 0.5 * (out + out.conj().T)      # Hermitian 化 (数值对称)
    return out * AU2CM


def so_coupled_energies(energies_cm: np.ndarray, soc_cm: np.ndarray
                        ) -> np.ndarray:
    """精细结构分辨耦合势: 对角化 ``diag(E) + H_SO``, 返回升序本征值 (cm⁻¹)。

    典型用法 (SOC-PES): 每个几何点传入无 SOC 的态能量与该点的 SOC 矩阵,
    得到精细结构分辨的势能曲线。
    """
    h = np.diag(np.asarray(energies_cm, dtype=float)) + np.asarray(soc_cm)
    ev = np.linalg.eigvalsh(h)
    return np.sort(ev)


def spin_pure_solver(ss: float, nroots: int, norb: int):
    """构造自旋纯化的 FCI 求解器 (``fci.addons.fix_spin_``)。

    ``ss`` = S(S+1): 0=单重态, 2=三重态, 6=五重态…
    """
    from pyscf import fci
    solver = fci.direct_spin1.FCISolver()
    solver.nroots = int(nroots)
    solver.max_cycle = 500
    fci.addons.fix_spin_(solver, shift=0.5, ss=float(ss))
    return solver


def casci_ci_vectors(mf, mo_coeff, ncas: int, nelecas, ss: float = 0.0,
                     nroots: int = 1):
    """在给定轨道上做 CASCI, 返回 ``(mc, energies_Ha, [ci...])`` —— **自旋纯**。

    自旋纯化双保险 (实测教训: 仅靠 ``fix_spin_`` 在部分扇区/初始猜测下会滞留
    在错误自旋上 —— 例如 CH₂ (1,1) 扇区请求三重态却返回带惩罚的单重态):
      1. ``fix_spin_`` 惩罚项;
      2. 按 ``<S²>`` 过滤 (必要时自动扩大 nroots 重求)。

    ``nelecas`` 请显式给 ``(na, nb)`` 元组以控制 M_s 分量 (如三重态 M=+1:
    (nα+1, nβ−1))。
    """
    from pyscf import mcscf
    from pyscf.fci import spin_op
    need = max(1, int(nroots))
    mc = mcscf.CASCI(mf, int(ncas), nelecas)
    mc.verbose = 0

    def _extract(obj):
        raw = obj.ci
        if isinstance(raw, (list, tuple)):
            ci = [np.asarray(c) for c in raw]
        else:
            a = np.asarray(raw)
            ci = [a] if a.ndim == 2 else [np.asarray(x) for x in a]
        return [c for c in ci if c.ndim == 2]

    last_err = None
    for n_r in (need, max(4 * need, 8), max(16 * need, 24)):
        mc.fcisolver = spin_pure_solver(ss, n_r, int(ncas))
        mc.kernel(np.asarray(mo_coeff))
        e = np.atleast_1d(np.asarray(mc.e_tot, dtype=float)).ravel()
        ci = _extract(mc)
        if not ci:
            last_err = "mc.ci 为空"
            continue
        s2 = []
        for c in ci:
            try:
                s2.append(float(spin_op.spin_square(c, int(ncas), nelecas)[0]))
            except Exception:
                s2.append(float("nan"))
        if ss >= 0:
            keep = [k for k, v in enumerate(s2) if abs(v - ss) < 0.1]
        else:
            keep = list(range(len(ci)))
        if len(keep) >= need:
            ci = [ci[k] for k in keep[:need]]
            e = np.asarray([e[k] for k in keep[:need]], dtype=float)
            return mc, e, ci
        if keep:                                  # 数量不足但至少有纯态 → 返回
            ci = [ci[k] for k in keep]
            e = np.asarray([e[k] for k in keep], dtype=float)
            return mc, e, ci
        last_err = f"<S²> = {np.round(s2, 4).tolist()} 均不匹配目标 S(S+1)={ss}"
    raise RuntimeError(
        f"CASCI 自旋纯化失败 (nelec={nelecas}, 目标 S(S+1)={ss}): {last_err}")
