import shutil
import unittest
import numpy as np

from aqpes.pes.calculators import (
    Calculator, AnalyticCalculator, demo_calculator,
    available_calculators, make_calculator, CommandBackendError,
)


class TestAnalyticBackend(unittest.TestCase):
    def test_analytic_gradient_vs_finite_difference(self):
        # 解析梯度与中心差分一致
        eps = 0.01, 0.0, 0.0
        coords = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        calc = demo_calculator()
        g = calc.gradient(coords)
        for i in range(2):
            for j in range(3):
                cp, cm = coords.copy(), coords.copy()
                cp[i, j] += 1e-6
                cm[i, j] -= 1e-6
                num = (calc.energy(cp) - calc.energy(cm)) / 2e-6
                self.assertAlmostEqual(g[i, j], num, places=5)

    def test_fd_fallback_matches_analytic(self):
        # 未提供解析梯度时中心差分回退应逼近解析梯度
        calc = demo_calculator()
        fd = AnalyticCalculator(calc.energy_fn, gradient_fn=None, grad_h=1e-5)
        coords = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
        np.testing.assert_allclose(fd.gradient(coords), calc.gradient(coords),
                                   rtol=1e-5, atol=1e-7)

    def test_energy_and_gradient(self):
        calc = demo_calculator()
        r_min = 2 ** (1 / 6) * 3.4      # LJ 极小点 → E = -ε
        coords = np.array([[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]])
        e, g = calc.energy_and_gradient(coords)
        self.assertAlmostEqual(e, -0.01, places=6)
        self.assertEqual(g.shape, coords.shape)
        # 极小点处力为零
        np.testing.assert_allclose(g, 0.0, atol=1e-6)
        # 力的反作用: ∇_A E = -∇_B E
        far = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        g2 = calc.gradient(far)
        np.testing.assert_allclose(g2[0], -g2[1], rtol=1e-12)
        self.assertEqual(calc.provenance["backend"], "demo-lj")


class TestBackendRegistry(unittest.TestCase):
    def test_available_calculators_reports_status(self):
        status = available_calculators()
        for key in ("analytic", "demo", "xtb", "pyscf", "ase"):
            self.assertIn(key, status)
            self.assertIn(status[key]["available"], ("yes", "no"))
        self.assertEqual(status["xtb"]["available"],
                         "yes" if shutil.which("xtb") else "no")

    def test_make_calculator_demo_and_unknown(self):
        calc = make_calculator("demo")
        self.assertIsInstance(calc, Calculator)
        with self.assertRaises(ValueError):
            make_calculator("no-such-backend")

    def test_xtb_missing_gives_informative_error(self):
        if shutil.which("xtb"):
            self.skipTest("xtb 已安装, 错误路径不可测")
        with self.assertRaises(CommandBackendError) as ctx:
            make_calculator("xtb", symbols=["H", "H"])
        self.assertIn("xtb", str(ctx.exception))


class TestPySCFCalculatorMock(unittest.TestCase):
    """测试 PySCFCalculator 的高自旋约束、spin_lock、MOM 及复共振势能接口 (Mock 隔离)。"""

    def _get_mock_modules(self):
        import sys
        from unittest.mock import MagicMock
        mock_pyscf = MagicMock()
        mock_pyscf.__version__ = "2.4.0"
        mock_gto = MagicMock()
        mock_scf = MagicMock()
        mock_dft = MagicMock()
        mock_addons = MagicMock()
        mock_lib = MagicMock()
        mock_lib.asarray.side_effect = lambda x: np.array(x)

        mock_pyscf.gto = mock_gto
        mock_pyscf.scf = mock_scf
        mock_scf.addons = mock_addons
        mock_pyscf.dft = mock_dft
        mock_pyscf.lib = mock_lib

        mods = {
            "pyscf": mock_pyscf,
            "pyscf.gto": mock_gto,
            "pyscf.scf": mock_scf,
            "pyscf.scf.addons": mock_addons,
            "pyscf.dft": mock_dft,
            "pyscf.lib": mock_lib,
        }
        return mods, mock_pyscf

    def test_pyscf_missing_gives_informative_error(self):
        import sys
        from unittest.mock import patch
        with patch.dict(sys.modules, {"pyscf": None}):
            with self.assertRaises(CommandBackendError) as ctx:
                make_calculator("pyscf", symbols=["He", "Li"])
            self.assertIn("未安装 pyscf", str(ctx.exception))

    def test_rhf_energy_and_gradient(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.235
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.RHF.return_value = mock_mf

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["He", "Li"], basis="def2-svp", method="rhf")
            coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
            e, g = calc.energy_and_gradient(coords)

            self.assertAlmostEqual(e, -7.235)
            self.assertEqual(g.shape, (2, 3))
            self.assertAlmostEqual(g[0, 0], 0.01)
            self.assertEqual(calc.provenance["method"], "rhf")
            self.assertEqual(calc.provenance["basis"], "def2-svp")

    def test_high_spin_rohf_and_uhf(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.150
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.ROHF.return_value = mock_mf

        with patch.dict(sys.modules, mods):
            # 四重态 (He* + Li, S=3/2, 2S=3)
            calc = make_calculator("pyscf", symbols=["He", "Li"], spin=3, method="rohf")
            coords = np.array([[0.0, 0.0, 0.0], [4.5, 0.0, 0.0]])
            e = calc.energy(coords)
            self.assertAlmostEqual(e, -7.150)
            mock_pyscf.scf.ROHF.assert_called_once()
            self.assertEqual(calc.provenance["spin"], "3")

    def test_spin_lock_audit_pass_and_fail(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.120
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.UHF.return_value = mock_mf

        coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        with patch.dict(sys.modules, mods):
            # 期望 S=1.5, S(S+1)=3.75
            # 情况 1: 自旋污染在容差内 (3.77 vs 3.75, diff=0.02 <= 0.1) -> 成功
            mock_mf.spin_square.return_value = (3.77, 1.506)
            calc_pass = make_calculator("pyscf", symbols=["He", "Li"], spin=3,
                                        method="uhf", spin_lock=True, spin_tol=0.1)
            e = calc_pass.energy(coords)
            self.assertAlmostEqual(e, -7.120)

            # 情况 2: 自旋污染超标 / 态翻转 (2.50 vs 3.75, diff=1.25 > 0.1) -> 触发拦截
            mock_mf.spin_square.return_value = (2.50, 1.1)
            calc_fail = make_calculator("pyscf", symbols=["He", "Li"], spin=3,
                                        method="uhf", spin_lock=True, spin_tol=0.1)
            with self.assertRaises(CommandBackendError) as ctx:
                calc_fail.energy(coords)
            self.assertIn("自旋锁定失败", str(ctx.exception))
            self.assertIn("自旋污染偏差", str(ctx.exception))

    def test_dft_and_xc_dispatch(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.300
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.dft.UKS.return_value = mock_mf

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["He", "Li"], spin=1,
                                   method="uks", xc="pbe")
            coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
            e = calc.energy(coords)
            self.assertAlmostEqual(e, -7.300)
            mock_pyscf.dft.UKS.assert_called_once()
            self.assertEqual(mock_mf.xc, "pbe")
            self.assertEqual(calc.provenance["xc"], "pbe")

    def test_mom_tracks_reference_orbitals(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol

        # 模拟两步 ROHF 计算 (He*+Li 四重态: 5 电子 = 4α + 1β, 2S=3)
        fake_mo_1 = np.eye(5)
        fake_occ_1 = np.array([2.0, 1.0, 1.0, 1.0, 0.0])  # ROHF 一维占据

        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.100
        mock_mf.mo_coeff = fake_mo_1
        mock_mf.mo_occ = fake_occ_1
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.ROHF.return_value = mock_mf
        mock_pyscf.scf.addons.mom_occ.return_value = mock_mf

        coords_1 = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        coords_2 = np.array([[0.0, 0.0, 0.0], [4.2, 0.0, 0.0]])

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["He", "Li"], spin=3,
                                   method="rohf", use_mom=True)
            # 第 1 步: 无先验参考, mom_occ 不应调用
            calc.energy(coords_1)
            mock_pyscf.scf.addons.mom_occ.assert_not_called()

            # 第 2 步: 存在第 1 步轨道, mom_occ 必须被注入, 且 setocc 为
            # (2, nmo) 的 alpha/beta 分离数组 (PySCF ROHF MOM 约定)
            calc.energy(coords_2)
            mock_pyscf.scf.addons.mom_occ.assert_called_once()
            call_args = mock_pyscf.scf.addons.mom_occ.call_args
            occorb_arg, setocc_arg = call_args[0][1], call_args[0][2]
            np.testing.assert_allclose(occorb_arg, fake_mo_1)
            self.assertEqual(setocc_arg.shape, (2, 5))
            # occ=[2,1,1,1,0] → alpha=[1,1,1,1,0], beta=[1,0,0,0,0]
            np.testing.assert_allclose(setocc_arg, [[1.0, 1.0, 1.0, 1.0, 0.0],
                                                    [1.0, 0.0, 0.0, 0.0, 0.0]])
            # 4α - 1β = 3 = 2S ✓, 总电子数 5 ✓
            self.assertAlmostEqual(setocc_arg[0].sum() - setocc_arg[1].sum(), 3.0)
            self.assertAlmostEqual(setocc_arg.sum(), 5.0)

            # 重置 MOM
            calc.reset_mom()
            self.assertIsNone(calc._ref_mo_coeff)

    def test_mom_setocc_conversion_uhf(self):
        """UHF/UKS 的 (2,nmo) 0/1 占据数组应原样透传。"""
        from aqpes.pes.calculators import PySCFCalculator
        occ_uhf = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        out = PySCFCalculator._mom_setocc(occ_uhf)
        np.testing.assert_allclose(out, occ_uhf)
        # ROHF 一维 {2,1,0} → (2,nmo)
        out_rohf = PySCFCalculator._mom_setocc(np.array([2.0, 1.0, 0.0]))
        np.testing.assert_allclose(out_rohf, [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    def test_resonance_width_and_complex_energy(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = self._get_mock_modules()

        mock_mol = MagicMock()
        mock_pyscf.gto.Mole.return_value = mock_mol
        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = -7.000
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.RHF.return_value = mock_mf

        coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])

        with patch.dict(sys.modules, mods):
            # 1. 默认无 CAP: 衰减宽度为 0, 复能量虚部为 0
            calc_real = make_calculator("pyscf", symbols=["He", "Li"])
            self.assertAlmostEqual(calc_real.resonance_width(coords), 0.0)
            self.assertEqual(calc_real.complex_energy(coords), complex(-7.000, 0.0))

            # 2. 指数型自电离宽度: Gamma(R) = A * exp(-beta * R)
            cap_exp = {"type": "exponential", "A": 0.05, "beta": 0.5, "r_index": (0, 1)}
            calc_exp = make_calculator("pyscf", symbols=["He", "Li"], cap_params=cap_exp)
            expected_gamma = 0.05 * np.exp(-0.5 * 4.0)
            self.assertAlmostEqual(calc_exp.resonance_width(coords), expected_gamma, places=7)
            z = calc_exp.complex_energy(coords)
            self.assertAlmostEqual(z.real, -7.000)
            self.assertAlmostEqual(z.imag, -0.5 * expected_gamma)

            # 3. 盒式 CAP 宽度: r_cap=3.0, R=4.0 > 3.0 -> Gamma = 2 * eta * (4 - 3)^2
            cap_box = {"type": "box", "eta": 0.01, "r_cap": 3.0, "r_index": (0, 1)}
            calc_box = make_calculator("pyscf", symbols=["He", "Li"], cap_params=cap_box)
            self.assertAlmostEqual(calc_box.resonance_width(coords), 2.0 * 0.01 * (1.0 ** 2))


class TestSACASSCFMock(unittest.TestCase):
    """态平均 CASSCF (v0.32.0): 分发 / 态选择 / NEVPT2(root) / 错误路径 (Mock)。"""

    def _mods(self):
        import sys
        from unittest.mock import MagicMock
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_mcscf = MagicMock()
        mock_mrpt = MagicMock()
        mock_solvent = MagicMock()
        mock_pyscf.mcscf = mock_mcscf
        mock_pyscf.mrpt = mock_mrpt
        mock_pyscf.solvent = mock_solvent
        mods.update({"pyscf.mcscf": mock_mcscf, "pyscf.mrpt": mock_mrpt,
                     "pyscf.solvent": mock_solvent})
        return mods, mock_pyscf, mock_mcscf, mock_mrpt

    def _mock_mc(self, mock_pyscf, mock_mcscf, e_states=(-1.10, -1.05)):
        from unittest.mock import MagicMock
        mock_pyscf.gto.Mole.return_value = MagicMock()
        mf = MagicMock()
        mf.converged = True
        mf.kernel.return_value = -1.12
        mf.mo_coeff = np.eye(4)
        mf.mo_occ = np.array([2.0, 0.0, 0.0, 0.0])
        mock_pyscf.scf.RHF.return_value = mf
        mc = MagicMock()
        mc.converged = True
        mc.kernel.return_value = (float(np.mean(e_states)), None)
        mc.e_states = np.asarray(e_states, dtype=float)
        mc.e_tot = float(np.mean(e_states))
        mc.ci = None
        mock_mcscf.CASSCF.return_value = mc
        return mc

    def test_state_average_dispatch_and_state_selection(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, _ = self._mods()
        mc = self._mock_mc(mock_pyscf, mock_mcscf)
        mock_mcscf.state_average_.return_value = mc

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                   active_space=(2, 2), nstates=2, state=1,
                                   state_average=True)
            e = calc.energy(np.array([[0., 0., 0.], [0., 0., 1.4]]))
            # 态平均必须被调用, 且等权 [0.5, 0.5]
            mock_mcscf.state_average_.assert_called_once()
            w = mock_mcscf.state_average_.call_args.args[1]
            np.testing.assert_allclose(w, [0.5, 0.5])
            # state=1 → 取第 2 个态的能量 (而非态平均能量)
            self.assertAlmostEqual(e, -1.05)
            np.testing.assert_allclose(calc.last_state_energies, [-1.10, -1.05])
            self.assertEqual(calc.provenance["nstates"], "2")
            self.assertEqual(calc.provenance["state"], "1")

    def test_nevpt2_on_excited_root(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, mock_mrpt = self._mods()
        mc = self._mock_mc(mock_pyscf, mock_mcscf)
        mock_mcscf.state_average_.return_value = mc
        nev = mock_mrpt.NEVPT.return_value
        nev.kernel.return_value = -0.02

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                   active_space=(2, 2), nstates=2, state=1,
                                   state_average=True, pt2="nevpt2")
            e = calc.energy(np.array([[0., 0., 0.], [0., 0., 1.4]]))
            # NEVPT2 必须落在所选的态 (root=1) 上
            self.assertEqual(mock_mrpt.NEVPT.call_args.kwargs.get("root"), 1)
            self.assertAlmostEqual(e, -1.05 - 0.02)

    def test_custom_state_weights(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, _ = self._mods()
        mc = self._mock_mc(mock_pyscf, mock_mcscf, e_states=(-1.10, -1.0, -0.9))
        mock_mcscf.state_average_.return_value = mc
        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                   active_space=(2, 2), nstates=3, state=0,
                                   state_average=True,
                                   state_weights=(0.6, 0.3, 0.1))
            calc.energy(np.array([[0., 0., 0.], [0., 0., 1.4]]))
            w = mock_mcscf.state_average_.call_args.args[1]
            np.testing.assert_allclose(w, [0.6, 0.3, 0.1])

    def test_error_paths(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, _ = self._mods()
        mc = self._mock_mc(mock_pyscf, mock_mcscf)
        mock_mcscf.state_average_.return_value = mc
        coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
        with patch.dict(sys.modules, mods):
            # 未开启态平均时 state=k>0 → 明确报错
            c1 = make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                 active_space=(2, 2), state=1)
            with self.assertRaises(CommandBackendError):
                c1.energy(coords)
            # 未开启态平均时 nstates>1 也**不得**隐式触发 (向后兼容)
            c1b = make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                  active_space=(2, 2), nstates=5)
            c1b.energy(coords)
            mock_mcscf.state_average_.assert_not_called()
            # state 越界
            c2 = make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                 active_space=(2, 2), nstates=2, state=5,
                                 state_average=True)
            with self.assertRaises(CommandBackendError):
                c2.energy(coords)
            # 权重长度不匹配
            c3 = make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                 active_space=(2, 2), nstates=2,
                                 state_average=True, state_weights=(1.0,))
            with self.assertRaises(CommandBackendError):
                c3.energy(coords)
            # 自旋纯求解器必须被装入 (fix_spin_)
            self.assertTrue(hasattr(mc, "fcisolver"))

    def test_fd_sampling_does_not_pollute_root_tracking(self):
        """_energy_only (FD 采样) 不得更新根跟踪参考态。"""
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_solvent = MagicMock()
        mock_pyscf.solvent = mock_solvent
        mods["pyscf.solvent"] = mock_solvent
        mock_pyscf.gto.Mole.return_value = MagicMock()
        mf = MagicMock()
        mf.converged = True
        mf.e_tot = -76.0
        td = MagicMock()
        td.nstates = 2
        td.kernel.return_value = (np.array([0.3, 0.5]), None)
        td.xy = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        td.oscillator_strength.return_value = np.array([0.1, 0.2])
        mock_pyscf.scf.RHF.return_value = mf
        mock_pyscf.tdscf = MagicMock()
        mock_pyscf.tdscf.rhf = MagicMock()
        mock_pyscf.tdscf.rhf.TDHF.return_value = td
        mods["pyscf.tdscf"] = mock_pyscf.tdscf
        mods["pyscf.tdscf.rhf"] = mock_pyscf.tdscf.rhf

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                   method="tddft", nstates=2, state=0,
                                   follow=True)
            coords = np.array([[0., 0., 0.], [0., 0., 1.4], [0., 1.0, -0.4]])
            calc.energy(coords)                 # 中心点: 建立参考态
            ref = None if calc._prev_exc_vec is None else calc._prev_exc_vec.copy()
            self.assertIsNotNone(ref)
            calc._energy_only(coords + 0.001)   # FD 采样
            np.testing.assert_allclose(calc._prev_exc_vec, ref)   # 参考态未被污染
            self.assertTrue(calc.follow)        # follow 标志已恢复


class TestSelectedCIMock(unittest.TestCase):
    """选择组态 CI (fci_solver="sci", v0.33.0) 的分发与参数落到求解器上。"""

    def _mods(self):
        import sys
        from unittest.mock import MagicMock
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_mcscf = MagicMock()
        mock_fci = MagicMock()
        mock_fci.addons = MagicMock()
        mock_sci_mod = MagicMock()
        mock_fci.selected_ci = mock_sci_mod
        mock_solvent = MagicMock()
        mock_pyscf.mcscf = mock_mcscf
        mock_pyscf.fci = mock_fci
        mock_pyscf.solvent = mock_solvent
        mods.update({"pyscf.mcscf": mock_mcscf, "pyscf.fci": mock_fci,
                     "pyscf.fci.addons": mock_fci.addons,
                     "pyscf.fci.selected_ci": mock_sci_mod,
                     "pyscf.solvent": mock_solvent})
        return mods, mock_pyscf, mock_mcscf, mock_sci_mod, mock_fci

    def _wire(self, mock_pyscf, mock_mcscf):
        from unittest.mock import MagicMock
        mock_pyscf.gto.Mole.return_value = MagicMock()
        mf = MagicMock()
        mf.converged = True
        mf.kernel.return_value = -107.0
        mf.mo_coeff = np.eye(6)
        mf.mo_occ = np.array([2., 2., 2., 0., 0., 0.])
        mf.mol = MagicMock()
        mock_pyscf.scf.RHF.return_value = mf
        mc = MagicMock()
        mc.converged = True
        mc.kernel.return_value = (-107.1, None)
        mc.e_tot = -107.1
        mc.e_states = np.array([-107.1, -107.05])
        mc.ci = None
        mock_mcscf.CASSCF.return_value = mc
        mock_mcscf.CASCI.return_value = mc
        mock_mcscf.state_average_.return_value = mc
        return mc

    def test_sci_solver_constructed_with_cutoffs(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, mock_sci_mod, mock_fci = self._mods()
        self._wire(mock_pyscf, mock_mcscf)
        with patch.dict(sys.modules, mods):
            # 大活性空间走 method="casci" (CASSCF 驱动与 SCI 的 RDM 接口不兼容)
            calc = make_calculator("pyscf", symbols=["N", "N"], method="casci",
                                   active_space=(6, 6), fci_solver="sci",
                                   sci_select_cutoff=1e-7,
                                   sci_ci_coeff_cutoff=1e-9)
            e = calc.energy(np.array([[0., 0., 0.], [0., 0., 2.1]]))
            self.assertAlmostEqual(e, -107.1)
            mock_sci_mod.SCI.assert_called_once()
            solver = mock_sci_mod.SCI.return_value
            self.assertAlmostEqual(solver.select_cutoff, 1e-7)
            self.assertAlmostEqual(solver.ci_coeff_cutoff, 1e-9)
            self.assertEqual(solver.nroots, 1)          # nroots 必须设为属性
            self.assertEqual(calc.provenance["fci_solver"], "sci")
            self.assertIn("select=1e-07", calc.provenance["sci_cutoffs"])

    def test_dense_default_and_spin_pure_toggle(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, mock_sci_mod, mock_fci = self._mods()
        self._wire(mock_pyscf, mock_mcscf)
        with patch.dict(sys.modules, mods):
            c1 = make_calculator("pyscf", symbols=["N", "N"], method="casscf",
                                 active_space=(6, 6))
            c1.energy(np.array([[0., 0., 0.], [0., 0., 2.1]]))
            mock_sci_mod.SCI.assert_not_called()             # 默认稠密
            mock_fci.direct_spin1.FCI.assert_called_once()
            # 单根默认**不**自旋纯化 (保持 v0.25–v0.32 历史行为)
            mock_fci.addons.fix_spin_.assert_not_called()
        for kw, expect in ((dict(spin_pure_fci=True), True),
                           (dict(nstates=2, state_average=True), True),
                           (dict(spin_pure_fci=False), False)):
            mods2, mock_pyscf2, mock_mcscf2, _, mock_fci2 = self._mods()
            self._wire(mock_pyscf2, mock_mcscf2)
            with patch.dict(sys.modules, mods2):
                c2 = make_calculator("pyscf", symbols=["N", "N"],
                                     method="casscf", active_space=(6, 6), **kw)
                c2.energy(np.array([[0., 0., 0.], [0., 0., 2.1]]))
                if expect:
                    self.assertTrue(mock_fci2.addons.fix_spin_.called, kw)
                else:
                    mock_fci2.addons.fix_spin_.assert_not_called()

    def test_casscf_with_sci_rejected(self):
        """CASSCF 驱动与 SCI 的 RDM 接口不兼容 → 必须明确报错并指向 casci。"""
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, mock_sci_mod, _ = self._mods()
        self._wire(mock_pyscf, mock_mcscf)
        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["N", "N"], method="casscf",
                                   active_space=(6, 6), fci_solver="sci")
            with self.assertRaises(CommandBackendError) as ctx:
                calc.energy(np.array([[0., 0., 0.], [0., 0., 2.1]]))
            self.assertIn("casci", str(ctx.exception))

    def test_invalid_fci_solver_rejected(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_pyscf.gto.Mole.return_value = MagicMock()
        with patch.dict(sys.modules, mods):
            with self.assertRaises(ValueError):
                make_calculator("pyscf", symbols=["H", "H"], method="casscf",
                                active_space=(2, 2), fci_solver="hci")


class TestAVASMock(unittest.TestCase):
    """AVAS 自动活性空间 (v0.35.0): 分发 / 冲突与方法守卫 / 与 SCI 组合。"""

    def _mods(self):
        import sys
        from unittest.mock import MagicMock
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_mcscf = MagicMock()
        mock_avas = MagicMock()
        mock_mcscf.avas = mock_avas
        mock_solvent = MagicMock()
        mock_pyscf.mcscf = mock_mcscf
        mock_pyscf.solvent = mock_solvent
        mods.update({"pyscf.mcscf": mock_mcscf,
                     "pyscf.mcscf.avas": mock_avas,
                     "pyscf.solvent": mock_solvent})
        return mods, mock_pyscf, mock_mcscf, mock_avas

    def _wire(self, mock_pyscf, mock_mcscf, mock_avas, ncas=5, nelecas=6):
        from unittest.mock import MagicMock
        mock_pyscf.gto.Mole.return_value = MagicMock()
        mf = MagicMock()
        mf.converged = True
        mf.kernel.return_value = -76.0
        mf.mo_coeff = np.eye(24)
        mf.mo_occ = np.array([2.] * 5 + [0.] * 19)
        mf.mol = MagicMock()
        mock_pyscf.scf.RHF.return_value = mf
        mo_avas = np.eye(24)
        mock_avas.avas.return_value = (ncas, nelecas, mo_avas)
        mc = MagicMock()
        mc.converged = True
        # CASCI/CASSCF 的 kernel 返回 (能量, ci, ...); 多根时能量为数组
        mc.kernel.return_value = (np.array([-76.1, -76.0]), None)
        mc.e_tot = -76.1
        mc.ci = None
        mc.e_states = np.array([-76.1, -76.0])
        mock_mcscf.CASCI.return_value = mc
        mock_mcscf.CASSCF.return_value = mc
        mock_mcscf.state_average_.return_value = mc
        return mc, mf, mo_avas

    def test_avas_dispatch_space_and_kernel_mo(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, mock_avas = self._mods()
        mc, mf, mo_avas = self._wire(mock_pyscf, mock_mcscf, mock_avas)
        coords = np.array([[0., 0., 0.], [0., 0., 1.4], [0., 1.0, -0.4]])
        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                   basis="cc-pvdz", method="casci",
                                   avas="O 2p; H 1s", avas_threshold=0.3)
            e = calc.energy(coords)
            mock_avas.avas.assert_called_once()
            args, kwargs = mock_avas.avas.call_args
            # 标签被规范化为**列表** (PySCF 的 avas 只认列表/单标签/正则串;
            # ';'/',' 串会静默给 ncas=0, 实测标定)
            self.assertEqual(args[1], ["O 2p", "H 1s"])
            self.assertAlmostEqual(kwargs["threshold"], 0.3)
            # 活性空间必须取自 AVAS 返回值 (ncas=5, nelecas=6)
            self.assertEqual(mock_mcscf.CASCI.call_args.args[1:], (5, 6))
            # AVAS 轨道必须传给 kernel
            mc.kernel.assert_called_once()
            np.testing.assert_allclose(mc.kernel.call_args.args[0], mo_avas)
            self.assertAlmostEqual(e, -76.1)
            prov = calc.provenance["avas"]
            self.assertIn("O 2p", prov)          # 规范化后的标签列表
            self.assertIn("H 1s", prov)
            self.assertIn("ncas=5", prov)

    def test_avas_conflict_and_method_guard(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf, _, _ = self._mods()
        mock_pyscf.gto.Mole.return_value = MagicMock()
        with patch.dict(sys.modules, mods):
            # avas 与 active_space 同时给出 → 报错
            with self.assertRaises(ValueError):
                make_calculator("pyscf", symbols=["O", "H", "H"],
                                method="casscf", active_space=(4, 4),
                                avas="O 2p")
            # avas 用于非多参考方法 → 报错
            with self.assertRaises(ValueError):
                make_calculator("pyscf", symbols=["O", "H", "H"],
                                method="rhf", avas="O 2p")
            # 两者都不给 → 原有明确报错
            calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                   method="casci")
            with self.assertRaises(CommandBackendError):
                calc.energy(np.array([[0., 0., 0.], [0., 0., 1.4], [0., 1.0, -0.4]]))

    def test_avas_with_sci_and_state_average(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_mcscf, mock_avas = self._mods()
        self._wire(mock_pyscf, mock_mcscf, mock_avas)
        mock_sci = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
        mods["pyscf.fci"] = mock_sci
        mods["pyscf.fci.selected_ci"] = mock_sci.selected_ci
        mods["pyscf.fci.addons"] = mock_sci.addons
        coords = np.array([[0., 0., 0.], [0., 0., 2.1]])
        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["N", "N"], basis="cc-pvdz",
                                   method="casci", avas="N 2p",
                                   fci_solver="sci", nstates=2, state=1,
                                   state_average=True)
            calc.energy(coords)
            mock_avas.avas.assert_called_once()                 # AVAS 生效
            mock_sci.selected_ci.SCI.assert_called_once()       # SCI 生效
            self.assertEqual(mock_sci.selected_ci.SCI.return_value.nroots, 2)


if __name__ == "__main__":
    unittest.main()

class TestCorrelatedMethodDispatch(unittest.TestCase):
    """相关方法 (MP2/CCSD/CCSD(T)) 的分发与梯度级别 (Mock 隔离)。"""

    def test_unknown_method_rejected(self):
        import sys
        from unittest.mock import MagicMock, patch
        import numpy as np
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        with patch.dict(sys.modules, mods):
            with self.assertRaises(ValueError):
                make_calculator("pyscf", symbols=["H", "H"], method="mp3")

    def test_correlated_dispatch_and_grad_level(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_pyscf.gto.Mole.return_value = MagicMock()
        mock_mf = MagicMock(); mock_mf.converged = True
        mock_mf.e_tot = -1.128
        mock_mf.kernel.return_value = -1.128
        mock_pyscf.scf.RHF.return_value = mock_mf
        # CCSD 求解器
        mock_cc = MagicMock()
        mock_cc.kernel.return_value = (-0.035, None, None)
        mock_cc.ccsd_t.return_value = -0.0001
        mock_grad = MagicMock(); mock_grad.kernel.return_value = np.zeros((2, 3))
        mock_cc.nuc_grad_method.return_value = mock_grad
        mock_pyscf.cc.CCSD.return_value = mock_cc

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["H", "H"], method="ccsd(t)")
            coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
            e, g = calc.energy_and_gradient(coords)
            # 能量 = SCF + E_corr + (T)
            self.assertAlmostEqual(e, -1.128 - 0.035 - 0.0001, places=6)
            self.assertEqual(g.shape, (2, 3))
            # provenance 记录梯度级别
            self.assertEqual(calc.provenance["grad_t_mode"], "fd")
            self.assertEqual(calc.provenance["method"], "ccsd(t)")

    def test_energy_does_not_compute_gradient(self):
        """energy() 不应触发梯度 (CCSD(T) 的 (T) 梯度是 6N 次 CCSD(T))。"""
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_pyscf.gto.Mole.return_value = MagicMock()
        mock_mf = MagicMock(); mock_mf.converged = True
        mock_mf.e_tot = -1.128; mock_mf.kernel.return_value = -1.128
        mock_pyscf.scf.RHF.return_value = mock_mf
        mock_cc = MagicMock()
        mock_cc.kernel.return_value = (-0.035, None, None)
        mock_cc.ccsd_t.return_value = -0.0001
        mock_pyscf.cc.CCSD.return_value = mock_cc

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["H", "H"], method="ccsd(t)")
            coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
            e = calc.energy(coords)
            mock_cc.nuc_grad_method.assert_not_called()


class TestPySCFSolventModelsMock(unittest.TestCase):
    """溶剂模型分发 (ddcosmo/pcm/ddpcm/smd) 的 Mock 隔离测试 (v0.30.0)。"""

    def _get_mock_modules(self):
        from unittest.mock import MagicMock
        mods, mock_pyscf = TestPySCFCalculatorMock()._get_mock_modules()
        mock_solvent = MagicMock()
        mock_pyscf.solvent = mock_solvent
        mods["pyscf.solvent"] = mock_solvent
        for name in ("ddcosmo", "pcm", "ddpcm"):
            sub = MagicMock()
            setattr(mock_solvent, name, sub)
            mods[f"pyscf.solvent.{name}"] = sub
        # 真实 smd 模块只有 SCF 入口 (smd_for_scf) → 用 spec 复现该边界
        mock_smd = MagicMock(
            spec=["smd_for_scf", "SMD", "solvent_db", "LEBEDEV_ORDER"])
        mock_solvent.smd = mock_smd
        mods["pyscf.solvent.smd"] = mock_smd
        return mods, mock_pyscf, mock_solvent

    def _mock_mf(self, mock_pyscf, mock_solvent, e=-76.02):
        from unittest.mock import MagicMock
        mock_pyscf.gto.Mole.return_value = MagicMock()
        mock_mf = MagicMock()
        mock_mf.converged = True
        mock_mf.kernel.return_value = e
        mock_grad = MagicMock()
        mock_grad.kernel.return_value = np.array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]])
        mock_mf.nuc_grad_method.return_value = mock_grad
        mock_pyscf.scf.RHF.return_value = mock_mf
        # 溶剂入口返回同一个 mf (带 with_solvent), 模拟 PySCF 包装行为
        for fn in (mock_solvent.ddcosmo.ddcosmo_for_scf,
                   mock_solvent.pcm.pcm_for_scf,
                   mock_solvent.ddpcm.ddpcm_for_scf,
                   mock_solvent.smd.smd_for_scf):
            fn.return_value = mock_mf
        return mock_mf

    def test_pcm_dispatch_sets_variant_and_eps(self):
        import sys
        from unittest.mock import patch
        from aqpes.pes.calculators import _SOLVENT_EPS
        mods, mock_pyscf, mock_solvent = self._get_mock_modules()
        mock_mf = self._mock_mf(mock_pyscf, mock_solvent)

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                   basis="6-31g*", solvent="water",
                                   solvent_model="pcm", pcm_variant="C-PCM")
            coords = np.array([[0., 0., 0.], [0., 0., 1.4], [0., 1.0, -0.4]])
            calc.energy(coords)
            mock_solvent.pcm.pcm_for_scf.assert_called_once()
            # PCM 变体与介电常数都必须落到溶剂对象上
            ws = mock_mf.with_solvent
            self.assertEqual(ws.method, "C-PCM")
            self.assertAlmostEqual(ws.eps, _SOLVENT_EPS["water"])
            self.assertIn("pcm/C-PCM", calc.provenance["solvent"])
            self.assertIn(str(_SOLVENT_EPS["water"]), calc.provenance["solvent"])

    def test_ddpcm_uses_fd_gradient(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_solvent = self._get_mock_modules()
        mock_mf = self._mock_mf(mock_pyscf, mock_solvent)

        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                   solvent_eps=78.4, solvent_model="ddpcm")
            self.assertTrue(calc._solvent_fd_grad)
            coords = np.array([[0., 0., 0.], [0., 0., 1.4], [0., 1.0, -0.4]])
            calc.energy_and_gradient(coords)
            # FD 路径 = 每扰动点一次 SCF (3 原子 → 19 次), 而非 1 次解析梯度
            self.assertGreaterEqual(mock_solvent.ddpcm.ddpcm_for_scf.call_count, 7)
            # ddPCM 无解析溶剂梯度模块 → 不得调用 nuc_grad_method
            mock_mf.nuc_grad_method.assert_not_called()

    def test_analytic_solvent_gradients_kept_for_ddcosmo_pcm(self):
        import sys
        from unittest.mock import patch
        for model in ("ddcosmo", "pcm"):
            mods, mock_pyscf, mock_solvent = self._get_mock_modules()
            mock_mf = self._mock_mf(mock_pyscf, mock_solvent)
            with patch.dict(sys.modules, mods):
                calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                       solvent_eps=78.4, solvent_model=model)
                self.assertFalse(calc._solvent_fd_grad)
                coords = np.array([[0., 0., 0.], [0., 0., 1.4], [0., 1.0, -0.4]])
                calc.gradient(coords)
                mock_mf.nuc_grad_method.assert_called_once()

    def test_smd_requires_named_solvent(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_solvent = self._get_mock_modules()
        self._mock_mf(mock_pyscf, mock_solvent)
        with patch.dict(sys.modules, mods):
            with self.assertRaises(ValueError) as ctx:
                make_calculator("pyscf", symbols=["O", "H", "H"],
                                solvent_model="smd")
            self.assertIn("命名溶剂", str(ctx.exception))
            with self.assertRaises(ValueError) as ctx2:
                make_calculator("pyscf", symbols=["O", "H", "H"],
                                solvent="water", solvent_eps=78.4,
                                solvent_model="smd")
            self.assertIn("solvent_eps", str(ctx2.exception))

    def test_smd_post_scf_entry_rejected(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf, mock_solvent = self._get_mock_modules()
        self._mock_mf(mock_pyscf, mock_solvent)
        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                   solvent="water", solvent_model="smd")
            # 非 SCF 入口必须明确报错 (而非静默回退气相)
            with self.assertRaises(CommandBackendError) as ctx:
                calc._attach_solvent(MagicMock(), "post")
            self.assertIn("SCF", str(ctx.exception))
            # SCF 入口正常, 且溶剂对象由 SMD 构造 (含非静电项参数集)
            calc._attach_solvent(MagicMock(), "scf")
            self.assertEqual(
                mock_solvent.smd.SMD.call_args.kwargs["solvent"], "water")

    def test_smd_unknown_solvent_gives_informative_error(self):
        import sys
        from unittest.mock import MagicMock, patch
        mods, mock_pyscf, mock_solvent = self._get_mock_modules()
        self._mock_mf(mock_pyscf, mock_solvent)
        mock_solvent.smd.SMD.side_effect = RuntimeError("nosuch is not available in SMD")
        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["O", "H", "H"],
                                   solvent="nosuch", solvent_model="smd")
            with self.assertRaises(CommandBackendError) as ctx:
                calc._attach_solvent(MagicMock(), "scf")
            self.assertIn("SMD", str(ctx.exception))

    def test_invalid_model_and_variant_rejected(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_solvent = self._get_mock_modules()
        self._mock_mf(mock_pyscf, mock_solvent)
        with patch.dict(sys.modules, mods):
            with self.assertRaises(ValueError):
                make_calculator("pyscf", symbols=["H", "H"],
                                solvent="water", solvent_model="cosmo2")
            with self.assertRaises(ValueError):
                make_calculator("pyscf", symbols=["H", "H"],
                                solvent="water", solvent_model="pcm",
                                pcm_variant="IEFPCM")

    def test_no_solvent_keeps_gas_phase_path(self):
        import sys
        from unittest.mock import patch
        mods, mock_pyscf, mock_solvent = self._get_mock_modules()
        self._mock_mf(mock_pyscf, mock_solvent)
        with patch.dict(sys.modules, mods):
            calc = make_calculator("pyscf", symbols=["H", "H"], method="rhf")
            coords = np.array([[0., 0., 0.], [0., 0., 1.4]])
            calc.energy(coords)
            mock_solvent.ddcosmo.ddcosmo_for_scf.assert_not_called()
            self.assertEqual(calc.provenance["solvent"], "none")


if __name__ == "__main__":
    unittest.main()
