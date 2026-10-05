"""复合方法 (CBS 外推 + CCSD(T) 加和) 本地单元测试 (无 pyscf 依赖)。"""
import unittest
import numpy as np

from aqpes.pes.composite import (cbs_extrapolate_2p, cbs_hf_2p, cbs_hf_3p,
                                       composite_energy, make_energy_fn,
                                       DEFAULT_SCHEME)


class TestCbsExtrapolation(unittest.TestCase):
    def test_power_law_exact_reconstruction(self):
        # 合成序列 E(X) = lim + A·X^-3 必须被精确还原
        for lim, A in ((-76.3, 0.5), (-1.1, 0.02), (0.0, 1.0)):
            e1 = lim + A * 3.0 ** -3
            e2 = lim + A * 4.0 ** -3
            self.assertAlmostEqual(cbs_extrapolate_2p(3, e1, 4, e2, 3.0),
                                   lim, places=10)

    def test_power_law_other_powers(self):
        lim, A = -5.0, 0.3
        e1, e2 = lim + A * 2 ** -4, lim + A * 3 ** -4
        self.assertAlmostEqual(cbs_extrapolate_2p(2, e1, 3, e2, 4.0), lim,
                               places=10)

    def test_exponential_hf_3p_exact(self):
        # 三点指数外推: b 由三点方程解出 → 合成序列应被**精确**还原
        lim, A, b = -76.0, 0.05, 1.2
        es = [lim + A * np.exp(-b * x) for x in (3, 4, 5)]
        got = cbs_hf_3p(3, es[0], 4, es[1], 5, es[2])
        self.assertLess(abs(got - lim), 1e-6)

    def test_exponential_hf_2p(self):
        # 两点: b 欠定 → 只能近似 (文档已注明; 默认方案用三点)
        lim, A, b = -76.0, 0.05, 1.2
        e1, e2 = lim + A * np.exp(-b * 4), lim + A * np.exp(-b * 5)
        self.assertLess(abs(cbs_hf_2p(4, e1, 5, e2) - lim), 5e-4)

    def test_errors(self):
        with self.assertRaises(ValueError):
            cbs_extrapolate_2p(4, -1.0, 3, -1.1)
        with self.assertRaises(ValueError):
            cbs_hf_2p(5, -1.0, 5, -1.1)


class TestCompositeEnergy(unittest.TestCase):
    def _stub(self, table, calls):
        """能量存根: E(coords,basis,method) 按 (basis,method) 查表。"""
        def fn(coords, basis, method):
            calls.append((basis, method))
            return table[(basis, method)]
        return fn

    def test_recipe_sums_components(self):
        # 构造: HF 指数序列 / MP2 相关能 X^-3 序列 / δCCSD(T) 常数
        hf_lim, hf_A, hf_b = -76.0, 0.04, 1.0
        corr_lim, corr_A = -0.30, 0.02
        tab = {}
        for X, basis in ((3, "cc-pvtz"), (4, "cc-pvqz"), (5, "cc-pv5z")):
            tab[(basis, "rhf")] = hf_lim + hf_A * np.exp(-hf_b * X)
        for X, basis in ((3, "cc-pvtz"), (4, "cc-pvqz")):
            e_hf = hf_lim + hf_A * np.exp(-hf_b * X)
            tab[(basis, "mp2")] = e_hf + corr_lim + corr_A * X ** -3
        delta = -0.01
        tab[("cc-pvtz", "ccsd(t)")] = tab[("cc-pvtz", "mp2")] + delta
        calls = []
        out = composite_energy(self._stub(tab, calls), np.zeros((2, 3)),
                               verbose=False)
        self.assertAlmostEqual(out["delta_ccsdt"], delta, places=12)
        # 总能量 ≈ HF 极限 + 相关极限 + δ (外推近似 b → 容差 1e-3)
        self.assertAlmostEqual(out["total"], hf_lim + corr_lim + delta,
                               delta=1e-6)
        # 六次能量调用 (HF×3 + MP2×2 + CCSD(T)×1)
        self.assertEqual(len(calls), 6)

    def test_scheme_override(self):
        tab = {("cc-pvdz", "rhf"): -1.0, ("cc-pvtz", "rhf"): -1.05,
               ("cc-pvdz", "mp2"): -1.1, ("cc-pvtz", "mp2"): -1.15}
        tab[("cc-pvdz", "ccsd")] = -1.12
        calls = []
        sch = dict(hf_bases=("cc-pvdz", "cc-pvtz"), hf_X=(2, 3),
                   corr_bases=("cc-pvdz", "cc-pvtz"), corr_X=(2, 3),
                   delta_basis="cc-pvdz", delta_method="ccsd",
                   ref_method="mp2")
        out = composite_energy(self._stub(tab, calls), np.zeros((2, 3)),
                               scheme=sch, verbose=False)
        self.assertIn("total", out)
        self.assertAlmostEqual(out["ccsdt_small"], -1.12)
        self.assertEqual(DEFAULT_SCHEME["delta_method"], "ccsd(t)")


if __name__ == "__main__":
    unittest.main()
