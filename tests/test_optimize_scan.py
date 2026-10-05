"""几何优化 / 谐振频率 / 内坐标扫描的解析模型单元测试 (无需 pyscf)。

用已知答案的模型势验证:
- 优化器能找到二次型极小;
- 质量加权 Hessian 与 cm⁻¹ 换算正确 (谐振子频率 √(k/μ));
- 键长扫描极小与解析 R_e 一致;
- 同位素质量表与约化质量自洽。
"""

import unittest
import numpy as np

from aqpes.pes.calculators import AnalyticCalculator
from aqpes.pes.optimize import (optimize_geometry, harmonic_frequencies,
                                      numerical_hessian)
from aqpes.pes import scan as Scan
from aqpes.pes.neb import neb_path, irc_path, _tangents
from aqpes.core.periodic import mass, AMU_TO_ME

BOHR = 0.529177210903
AU_FREQ_TO_CM = 219474.6313632


def harmonic_calc(k: float, r_e: float):
    """成对谐振子势 E = Σ_{i<j} k/2 (r_ij - r_e)^2 (任意原子数)。"""
    def energy(c):
        c = np.asarray(c, dtype=float)
        tot = 0.0
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                tot += 0.5 * k * (np.linalg.norm(c[i] - c[j]) - r_e) ** 2
        return float(tot)

    def grad(c):
        c = np.asarray(c, dtype=float)
        g = np.zeros_like(c)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                d = c[i] - c[j]
                r = float(np.linalg.norm(d))
                f = k * (r - r_e) * d / r
                g[i] += f
                g[j] -= f
        return g

    return AnalyticCalculator(energy, grad, name="harmonic")


def morse_calc(de: float, alpha: float, r_e: float):
    """成对 Morse 势 E = Σ_{i<j} D_e [1 - exp(-α(r_ij - r_e))]^2。"""
    def energy(c):
        c = np.asarray(c, dtype=float)
        tot = 0.0
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                r = float(np.linalg.norm(c[i] - c[j]))
                tot += de * (1.0 - np.exp(-alpha * (r - r_e))) ** 2
        return float(tot)

    def grad(c):
        c = np.asarray(c, dtype=float)
        g = np.zeros_like(c)
        for i in range(len(c)):
            for j in range(i + 1, len(c)):
                d = c[i] - c[j]
                r = float(np.linalg.norm(d))
                e = np.exp(-alpha * (r - r_e))
                f = 2.0 * de * (1.0 - e) * alpha * e * d / r
                g[i] += f
                g[j] -= f
        return g

    return AnalyticCalculator(energy, grad, name="morse")


class TestPeriodic(unittest.TestCase):
    def test_isotope_masses(self):
        self.assertAlmostEqual(mass("H"), 1.0078250319, places=9)
        self.assertAlmostEqual(mass("C"), 12.0, places=9)
        self.assertAlmostEqual(mass("O"), 15.99491461957, places=9)
        self.assertGreater(mass("D"), mass("H"))

    def test_unknown_element_raises(self):
        with self.assertRaises(KeyError):
            mass("Xx")
        # extra 表可覆盖
        self.assertAlmostEqual(mass("Xx", extra={"Xx": 5.0}), 5.0)


class TestOptimizer(unittest.TestCase):
    def test_finds_quadratic_minimum(self):
        k, r_e = 0.5, 1.4
        calc = harmonic_calc(k, r_e)
        x0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        opt, info = optimize_geometry(calc, x0, gtol=1e-8)
        r = float(np.linalg.norm(opt[0] - opt[1]))
        self.assertAlmostEqual(r, r_e, places=6)
        self.assertTrue(info["converged"])
        self.assertLess(info["grad_max"], 1e-6)
        self.assertAlmostEqual(info["energy"], 0.0, places=10)

    def test_morse_minimum_and_curvature(self):
        de, alpha, r_e = 0.2, 1.0, 2.0
        calc = morse_calc(de, alpha, r_e)
        x0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])
        opt, info = optimize_geometry(calc, x0, gtol=1e-8)
        self.assertAlmostEqual(float(np.linalg.norm(opt[0] - opt[1])), r_e,
                               places=6)
        # 极小处曲率 = 2 D_e α²
        H = numerical_hessian(calc, opt, h=1e-3)
        # 沿键方向的二阶导 (两原子相对位移)
        self.assertAlmostEqual(abs(H[2, 2]), 2 * de * alpha ** 2, delta=1e-3)

    def test_line_search_survives_bad_start(self):
        calc = morse_calc(0.1, 0.8, 2.0)
        x0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])   # 远离极小
        opt, info = optimize_geometry(calc, x0, gtol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(opt[0] - opt[1])), 2.0,
                               places=5)


class TestFrequencies(unittest.TestCase):
    def test_diatomic_harmonic_frequency(self):
        """ν = (1/2π)√(k/μ): 与解析值比较 (cm⁻¹)。"""
        k, r_e = 0.35, 1.4            # Hartree/Bohr²
        calc = harmonic_calc(k, r_e)
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_e]])
        freqs, info = harmonic_frequencies(calc, ["H", "H"], coords)
        self.assertEqual(len(freqs), 1)                 # 线性双原子: 3N-5 = 1
        mu = (mass("H") * mass("H") / (2 * mass("H"))) * AMU_TO_ME
        nu_ref = np.sqrt(k / mu) * AU_FREQ_TO_CM
        self.assertAlmostEqual(freqs[0], nu_ref, delta=1.0)
        self.assertEqual(info["n_imag"], 0)

    def test_isotope_shift(self):
        """同位素取代: ν ∝ 1/√μ (H2 -> D2 应下降 √2 倍)。"""
        k, r_e = 0.35, 1.4
        calc = harmonic_calc(k, r_e)
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, r_e]])
        f_h, _ = harmonic_frequencies(calc, ["H", "H"], coords)
        f_d, _ = harmonic_frequencies(calc, ["D", "D"], coords)
        mu_h = mass("H") / 2 * AMU_TO_ME
        mu_d = mass("D") / 2 * AMU_TO_ME
        self.assertAlmostEqual(f_d[0] / f_h[0], np.sqrt(mu_h / mu_d),
                               places=6)

    def test_saddle_reports_one_imaginary_mode(self):
        """鞍点必须报 1 个虚频 (取最大特征值的旧实现会漏掉负特征值)。"""
        calc = harmonic_calc(-0.4, 2.0)      # 负曲率 = 抛物线势垒
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        freqs, info = harmonic_frequencies(calc, ["H", "H"], coords)
        self.assertEqual(info["n_imag"], 1)
        self.assertEqual(len(freqs), 1)
        self.assertLess(freqs[0], 0.0)
        mu = (mass("H") / 2) * AMU_TO_ME
        self.assertAlmostEqual(freqs[0], -np.sqrt(0.4 / mu) * AU_FREQ_TO_CM,
                               delta=1.0)

    def test_nonlinear_water_like_modes(self):
        """非线性三原子: 应有 3N-6 = 3 个振动模式。"""
        calc = morse_calc(0.1, 1.0, 2.0)
        # 等边三角形 (边长 = r_e) 是成对 Morse 的极小
        a = 2.0
        coords = np.array([[0.0, 0.0, 0.0],
                           [a, 0.0, 0.0],
                           [a / 2, a * np.sqrt(3) / 2, 0.0]])
        freqs, info = harmonic_frequencies(calc, ["H", "H", "H"], coords)
        self.assertEqual(len(freqs), 3)
        self.assertEqual(info["n_imag"], 0)


class TestScan(unittest.TestCase):
    def test_bond_scan_minimum_matches_analytic(self):
        # 谐振子: 抛物拟合应精确复现 r_e
        calc_h = harmonic_calc(0.3, 2.3)
        grid = np.linspace(1.6, 3.2, 17)
        c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.3]])
        dh = Scan.scan_bond(calc_h, ["H", "H"], c0, 0, 1, grid)
        i = int(np.argmin(dh.energies))
        sl = slice(max(0, i - 2), min(len(grid), i + 3))
        coef = np.polyfit(grid[sl], dh.energies[sl], 2)
        self.assertAlmostEqual(-coef[1] / (2 * coef[0]), 2.3, places=6)
        # Morse: 非谐性使 5 点抛物拟合有 ~1% 偏差, 容差 0.03 Bohr
        calc = morse_calc(0.15, 1.1, 2.3)
        data = Scan.scan_bond(calc, ["H", "H"], c0, 0, 1, grid)
        i = int(np.argmin(data.energies))
        sl = slice(max(0, i - 2), min(len(grid), i + 3))
        coef = np.polyfit(grid[sl], data.energies[sl], 2)
        self.assertAlmostEqual(-coef[1] / (2 * coef[0]), 2.3, delta=0.03)
        # 数据容器完整 (含梯度与符号)
        self.assertEqual(data.points.shape, (17, 6))
        self.assertEqual(data.gradients.shape, (17, 2, 3))
        self.assertEqual(data.symbols, ["H", "H"])

    def test_angle_scan(self):
        calc = morse_calc(0.1, 1.0, 2.0)
        sym = ["H", "H", "H"]
        c0 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.6, 1.8, 0.0]])
        grid = np.linspace(60.0, 180.0, 13)
        data = Scan.scan_angle(calc, sym, c0, 1, 0, 2, grid)
        # 顶点 0: 扫描后角度应等于设定值
        for k, a in enumerate(grid):
            c = data.geometry[k]
            u, v = c[1] - c[0], c[2] - c[0]
            ang = np.degrees(np.arccos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))
            self.assertAlmostEqual(ang, a, places=6)

    def test_path_scan_endpoints(self):
        calc = morse_calc(0.1, 1.0, 2.0)
        a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])
        b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        data = Scan.scan_path(calc, ["H", "H"], a, b, 5)
        np.testing.assert_allclose(data.geometry[0], a)
        np.testing.assert_allclose(data.geometry[-1], b)
        self.assertEqual(data.points.shape[0], 5)

    def test_relaxed_scan_preserves_bond(self):
        """松弛扫描: 每点键长精确保持 (用模型势, 无需 pyscf)。"""
        calc = morse_calc(0.12, 1.0, 2.1)
        sym = ["H", "H"]
        c0 = np.array([[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
        grid = np.linspace(1.9, 2.3, 3)
        data = Scan.relaxed_scan_bond(calc, sym, c0, 0, 1, grid, gtol=1e-6)
        for k, r in enumerate(grid):
            got = float(np.linalg.norm(data.geometry[k][0] - data.geometry[k][1]))
            self.assertAlmostEqual(got, r, places=9)


class TestNEB(unittest.TestCase):
    """CI-NEB: 解析面 (LEPS H+H₂) 上的势垒与鞍点几何。

    本测试可直接捕获两类真实 bug:
    - 切向的负索引回绕 (i=0 用 ext[i-1] 会取到终点) → 路径塌陷;
    - 势垒参照错用第一个镜像而非反应物端点。
    """

    @staticmethod
    def _leps_calc():
        from aqpes.pes.leps import LEPSBuilder
        lb = LEPSBuilder()

        def energy(c):
            c = np.asarray(c, dtype=float)
            return float(np.ravel(lb.evaluate_2d(
                abs(c[0][0] - c[1][0]), abs(c[1][0] - c[2][0])))[0])

        def grad(c):
            c = np.asarray(c, dtype=float)
            h = 1e-6
            g = np.zeros_like(c)
            for i in range(3):
                for j in range(3):
                    cp, cm = c.copy(), c.copy()
                    cp[i, j] += h; cm[i, j] -= h
                    g[i, j] = (energy(cp) - energy(cm)) / (2 * h)
            return g

        return AnalyticCalculator(energy, grad, name="leps")

    def test_tangent_no_wraparound(self):
        """首末镜像的切向必须由固定端点定义 (不能被负索引回绕污染)。"""
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[10.0, 0.0, 0.0]])
        imgs = np.array([[[2.0, 0, 0]], [[5.0, 0, 0]], [[8.0, 0, 0]]])
        tau = _tangents(imgs, a, b)
        # 全部应指向 +x (路径单调)
        for i in range(3):
            self.assertGreater(tau[i][0, 0], 0.99)

    def test_leps_irc_symmetry_and_monotonicity(self):
        """IRC: 对称反应双方向应单调下降且结果镜像一致 (质量加权换算与
        方向符号的正确性检验; 写错会成为"停滞在 TS"或"飞出"两类故障)。"""
        calc = self._leps_calc()
        r_ts = 2.212
        ts = np.array([[0.0, 0.0, 0.0], [r_ts, 0.0, 0.0], [2 * r_ts, 0.0, 0.0]])
        res = {}
        for d in (+1, -1):
            path, info = irc_path(calc, ["H"] * 3, ts, step=0.08,
                                  max_steps=300, direction=d)
            e = info["energies"]
            frac = float(np.mean(np.diff(e) <= 1e-6))
            self.assertGreater(frac, 0.98)          # 单调 (无之字形)
            self.assertGreater(e[0] - e[-1], 0.03)  # 确实向下走了
            res[d] = (e[0] - e[-1], path[-1][:, 0].copy())
        # 对称反应: 两个方向的能量下降量应一致
        self.assertAlmostEqual(res[+1][0], res[-1][0], places=6)
        # 几何互为镜像: R01(+1) ≈ R12(-1)
        p1, m1 = res[+1][1], res[-1][1]
        r01_p = abs(p1[0] - p1[1]); r12_p = abs(p1[1] - p1[2])
        r01_m = abs(m1[0] - m1[1]); r12_m = abs(m1[1] - m1[2])
        self.assertAlmostEqual(r01_p, r12_m, delta=1e-6)

    def test_leps_barrier_and_saddle(self):
        from aqpes.pes.leps import LEPSBuilder
        calc = self._leps_calc()
        A = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [5.401, 0.0, 0.0]])
        B = np.array([[0.0, 0.0, 0.0], [1.401, 0.0, 0.0], [5.401, 0.0, 0.0]])
        lb = LEPSBuilder()
        rr = np.linspace(1.2, 4.0, 800)
        v = np.ravel(lb.evaluate_2d(rr, rr))
        e_react = float(np.ravel(lb.evaluate_2d(4.0, 1.401))[0])
        bar_ref = (v.min() - e_react) * 27.211386245988
        imgs, info = neb_path(calc, ["H"] * 3, A, B, n_images=9,
                              k_spring=0.08, max_iter=800, gtol=1e-2)
        # 势垒: 解析鞍点参照, 容差 5%
        self.assertAlmostEqual(info["barrier_eV"], bar_ref, delta=0.05 * bar_ref)
        # 鞍点几何: 对称 (R1 ≈ R2) 且对应解析值
        ts = imgs[info["ts_index"]]
        r1 = abs(ts[0][0] - ts[1][0]); r2 = abs(ts[1][0] - ts[2][0])
        self.assertLess(abs(r1 - r2) / max(r1, r2), 0.03)
        r_ref = float(rr[int(np.argmin(v))])
        self.assertAlmostEqual(0.5 * (r1 + r2), r_ref, delta=0.05)
        # 对称反应: ΔE_rxn = 0
        self.assertAlmostEqual(info["reaction_energy_eV"], 0.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
