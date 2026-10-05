"""SOC 模块本地单元测试 (无需安装 pyscf)。

覆盖:
  - soc_zeta 与精细结构分裂关系 (合成 h_mo)
  - so_coupled_energies 对角化
  - soc_integrals 的逐原子 Z 加权与 α²/2 前置因子 (mock mol)
  - trans_density_1body 行列式代数 (假 cistring, 手工可验证的小体系)
  - 费米子符号约定 (满壳层 α 串: D = [[0,-1],[1,0]])
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from aqpes.pes import soc as S


def _fake_modules():
    """纯 Python 版 cistring.gen_occslst (递增位串) 的模块树。

    注意: ``from pyscf.fci import cistring`` 会走 ``getattr(pyscf.fci, ...)``,
    故必须把子模块挂在父 mock 上 (仅放进 sys.modules 不够)。
    """
    from itertools import combinations
    cs = MagicMock()

    def gen_occslst(orbs, nocc):
        # 与真实 PySCF 一致: 返回占据轨道索引列表 (nstr, nocc)
        orbs = list(orbs)
        rows = [list(c) for c in combinations(orbs, nocc)]
        if int(nocc) == 0:
            return np.zeros((len(rows), 0), dtype=np.int64)
        return np.array(rows, dtype=np.int64)

    def strs2addr(norb, nocc, strs):
        # 本地参考实现: 地址 = 位串在递增排序中的位置
        all_s = gen_occslst(range(norb), nocc)
        bits = np.array([sum(1 << int(i) for i in row) for row in all_s],
                        dtype=np.int64)
        order = np.argsort(bits)
        table = {int(b): int(k) for k, b in enumerate(bits[order])}
        return np.array([table[int(x)] for x in np.asarray(strs).ravel()],
                        dtype=np.int64)

    cs.gen_occslst.side_effect = gen_occslst
    cs.strs2addr.side_effect = strs2addr
    fci = MagicMock()
    fci.cistring = cs
    root = MagicMock()
    root.fci = fci
    return {"pyscf": root, "pyscf.fci": fci, "pyscf.fci.cistring": cs}


class TestSocConstants(unittest.TestCase):
    def test_zeta_from_three_level_block(self):
        # h_z 在实基下应为纯虚本征值 (0, ±iζ)
        zeta = 0.0012            # Hartree
        blk = np.array([[0.0, -1j * zeta, 0.0],
                        [1j * zeta, 0.0, 0.0],
                        [0.0, 0.0, 0.0]])
        # 转为实表示: h = blk / 1j  (i·h 的本征值为 ±ζ)
        h_z_real = (blk / 1j).real
        h_mo = np.zeros((3, 3, 3))
        h_mo[2] = h_z_real
        got = S.soc_zeta(h_mo, [0, 1, 2])
        self.assertAlmostEqual(got, zeta * S.AU2CM, places=6)
        # 相位/混合无关: 对子空间做任意正交变换不改变 ζ
        rng = np.random.default_rng(3)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        h_rot = (q.T @ (blk / 1j) @ q).real
        h2 = np.zeros((3, 3, 3))
        h2[2] = h_rot
        self.assertAlmostEqual(S.soc_zeta(h2, [0, 1, 2]),
                               zeta * S.AU2CM, places=6)

    def test_splitting_relations(self):
        z = 100.0
        self.assertAlmostEqual(S.splitting_atomic_p(z), 150.0)
        self.assertAlmostEqual(S.splitting_atomic_p(-z), 150.0)
        self.assertAlmostEqual(S.splitting_pi(-z), 100.0)

    def test_coupled_energies(self):
        # 两个简并态 + 纯虚耦合 → 分裂 2|c|
        c = 50.0
        v = np.array([0.0, 0.0])
        soc = np.array([[0.0, 1j * c], [-1j * c, 0.0]])
        ev = S.so_coupled_energies(v, soc)
        np.testing.assert_allclose(ev, [-c, c], atol=1e-10)
        # 对角项只平移
        ev2 = S.so_coupled_energies(np.array([10.0, 20.0]), np.zeros((2, 2)))
        np.testing.assert_allclose(ev2, [10.0, 20.0], atol=1e-12)

    def test_soc_integrals_atom_weighting_and_prefactor(self):
        mol = MagicMock()
        mol.nao = 2
        mol.natm = 2
        mol.atom_symbol.side_effect = lambda i: ("H", "F")[i]
        mol.atom_charge.side_effect = lambda i: (1.0, 9.0)[i]
        mol.atom_coord.side_effect = lambda i: np.array([0.0, 0.0, float(i)])
        one = np.ones((3, 2, 2))
        mol.intor.side_effect = lambda name, comp=None: one
        out = S.soc_integrals(mol)
        # (1 + 9) × ones × α²/2 × (−i)  [物理 l = −i(r×∇)]
        np.testing.assert_allclose(out, -1j * 10.0 * S.PREF * one)
        # rinv 原点必须逐原子设置 (int1e_prinvxp 的 1/r³ 起点; 实测
        # with_common_origin 对该积分无效, 会使结果随分子平移漂移)
        self.assertEqual(mol.with_rinv_at_nucleus.call_count, 2)
        # z_eff 覆盖 F 的有效核电荷
        out2 = S.soc_integrals(mol, z_eff={"F": 2.0})
        np.testing.assert_allclose(out2, -1j * 3.0 * S.PREF * one)
        # 零电荷原子被跳过
        mol.atom_charge.side_effect = lambda i: (1.0, 0.0)[i]
        mol.with_rinv_at_nucleus.reset_mock()
        S.soc_integrals(mol)
        self.assertEqual(mol.with_rinv_at_nucleus.call_count, 1)


class TestTransitionDensity(unittest.TestCase):
    """行列式级跃迁密度: 用假 cistring 做手工可验证的小体系检验。"""

    def _run(self, fn):
        with patch.dict(sys.modules, _fake_modules()):
            return fn()

    def test_filled_shell_density_identity(self):
        # ket = |alpha_0 alpha_1> (norb=2, nelec=(2,0)) → D_aa = I (两轨道全占)
        def body():
            ci = np.array([[1.0]])          # 单行列式
            return S.trans_density_1body("aa", ci.ravel(), ci.ravel(), 2, (2, 0))
        d = self._run(body)
        np.testing.assert_allclose(d, np.eye(2), atol=1e-12)

    def test_transition_density_parity_sign(self):
        # 3 轨道 (2,0) 扇区: |a0 a1> → |a0 a2> 的跃迁密度只有 D[2,1] = +1
        # a_1|a0a1> = -|a0> (奇宇称); 再 a†_2(-|a0>) = -a†_2a†_0|0> = +|a0a2>
        # (a†_2a†_0 = -a†_0a†_2), 故矩阵元 = +1
        def body():
            ci_a = np.array([[1.0, 0.0, 0.0]])      # |a0 a1> (3 个行列式)
            ci_b = np.array([[0.0, 1.0, 0.0]])      # |a0 a2>
            return S.trans_density_1body("aa", ci_b.ravel(), ci_a.ravel(), 3, (2, 0))
        d = self._run(body)
        expect = np.zeros((3, 3))
        expect[2, 1] = 1.0
        np.testing.assert_allclose(d, expect, atol=1e-12)

    def test_single_particle_density(self):
        # ket = (|0> + |1>)/√2 (1 个 α 电子) → D = 1/2 * all-ones
        def body():
            ci = np.array([[1.0, 1.0]]) / np.sqrt(2.0)   # (nα=2, nβ=1) 串
            return S.trans_density_1body("aa", ci.ravel(), ci.ravel(), 2, (1, 0))
        d = self._run(body)
        np.testing.assert_allclose(d, np.full((2, 2), 0.5), atol=1e-12)

    def test_spinflip_density_outer_product(self):
        # ket: 1 个 β 电子 (nβ=1) 于轨道 0/1 叠加; bra: 1 个 α 电子 → D^ab = c_b c_k^T
        def body():
            ck = np.array([[1.0, 2.0]]) / np.sqrt(5.0)   # (nα=1, nβ=2) 串空间
            cb = np.array([[3.0, 1.0]]) / np.sqrt(10.0)
            return S.trans_density_1body("ab", cb.ravel(), ck.ravel(), 2, (0, 1))
        d = self._run(body)
        expected = np.outer([3.0, 1.0] / np.sqrt(10.0), [1.0, 2.0] / np.sqrt(5.0))
        np.testing.assert_allclose(d, expected, atol=1e-12)

    def test_spinflip_sign_with_occupied_string(self):
        # ket: 2 个 β 电子 |β_0 β_1>; bra: 1α1β. D^ab[p,q] = <bra|a†_pα a_qβ|ket>
        # a_0|β0β1> = +|β1>, a_1|β0β1> = -|β0>
        def body():
            ck = np.array([[1.0]])                        # (nα=0,nβ=2): |β0β1>
            cb = np.zeros((2, 2))                         # (nα=1,nβ=1)
            cb[0, 1] = 1.0                                # bra = |α0 β1>
            return S.trans_density_1body("ab", cb.ravel(), ck.ravel(), 2, (0, 2))
        d = self._run(body)
        # bra=|α0 β1> 与 a†_0α a_0β|ket> = |α0 β1> 匹配 (sign +)
        self.assertAlmostEqual(d[0, 0], 1.0, places=12)
        self.assertAlmostEqual(d[1, 0], 0.0, places=12)

    def test_soc_matrix_triplet_vector_relations(self):
        # 人为构造: 三个 M 分量与单重态的自旋翻转/zeeman 耦合必须等量 (WET)
        def body():
            hz = np.array([[0.0, 0.0], [0.0, 0.0]])
            hx = np.array([[0.0, 0.3], [0.3, 0.0]])
            hy = np.array([[0.0, 0.4], [0.4, 0.0]])
            h_mo = np.stack([hx, hy, hz])
            sing = {"ci": np.array([[1.0, 0.0], [0.0, 1.0]]).ravel(),
                    "nelec": (1, 1)}
            t0 = {"ci": np.array([[0.0, 1.0], [-1.0, 0.0]]).ravel(),
                  "nelec": (1, 1)}
            tp = {"ci": np.array([[1.0]]), "nelec": (2, 0)}
            tm = {"ci": np.array([[1.0]]), "nelec": (0, 2)}
            return S.soc_matrix_states(h_mo, [sing, t0, tp, tm])
        h = self._run(body)
        v = np.abs([h[0, 0], h[0, 1], h[0, 2], h[0, 3]])
        # 对角线 (单重态自身) 与 z 分量由对称性为零; 两个自旋翻转分量等量
        self.assertAlmostEqual(v[0], 0.0, places=10)
        self.assertAlmostEqual(v[1], 0.0, places=10)
        self.assertAlmostEqual(v[2], v[3], places=6)


if __name__ == "__main__":
    unittest.main()
