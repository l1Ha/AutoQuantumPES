import os
import shutil
import tempfile
import unittest
import numpy as np

from aqpes.pes.calculators import demo_calculator
from aqpes.pes.abinitio import AbInitioData


class TestGeometrySampling(unittest.TestCase):
    def _ref(self):
        return np.array([[0.0, 0.0, 0.0], [3.4, 0.0, 0.0]])

    def test_sample_shapes_and_consistency(self):
        calc = demo_calculator()
        data = AbInitioData.sample_geometries(
            calc, self._ref(), ranges=((-0.3, 0.3),), n_per_dim=5,
            active_atoms=[1], axes=(0,), min_distance=1.0)
        self.assertEqual(data.points.shape, (5, 6))          # (n, 3N)
        self.assertEqual(data.gradients.shape, (5, 2, 3))    # 几何梯度
        self.assertEqual(data.n_points, 5)
        self.assertEqual(data.provenance["backend"], "demo-lj")
        # 采样点与参考几何一致 (仅原子 1 沿 x 位移)
        np.testing.assert_allclose(
            data.points[:, :3],
            np.repeat(self._ref()[0][None, :], data.n_points, axis=0),
            atol=1e-12)
        # 能量与直接调用后端一致
        e_direct = calc.energy(data.geometry[2])
        self.assertAlmostEqual(float(data.energies[2]), e_direct, places=12)

    def test_min_distance_filter(self):
        calc = demo_calculator()
        data = AbInitioData.sample_geometries(
            calc, self._ref(), ranges=((-3.5, 0.5),), n_per_dim=9,
            active_atoms=[1], axes=(0,), min_distance=2.0)
        dists = np.linalg.norm(data.geometry[:, 0] - data.geometry[:, 1], axis=1)
        self.assertTrue(np.all(dists >= 2.0 - 1e-9))
        self.assertLess(data.n_points, 9)  # 近距离构型被剔除

    def test_ranges_broadcast_and_validation(self):
        calc = demo_calculator()
        AbInitioData.sample_geometries(   # 单元素 ranges 广播到 3 维
            calc, self._ref(), ranges=((-0.2, 0.2),), n_per_dim=3,
            active_atoms=[1], axes=(0, 1, 2), min_distance=1.0)
        with self.assertRaises(ValueError):
            AbInitioData.sample_geometries(
                calc, self._ref(), ranges=((-0.2, 0.2), (0.0, 0.0)),
                n_per_dim=3, active_atoms=[1], axes=(0, 1, 2))

    def test_max_points_subsampling(self):
        calc = demo_calculator()
        data = AbInitioData.sample_geometries(
            calc, self._ref(), ranges=((-0.5, 0.5),), n_per_dim=9,
            active_atoms=[1], axes=(0,), min_distance=1.0, max_points=5)
        self.assertEqual(data.n_points, 5)

    def test_npz_roundtrip_with_provenance(self):
        calc = demo_calculator()
        data = AbInitioData.sample_geometries(
            calc, self._ref(), ranges=((-0.2, 0.2),), n_per_dim=4,
            active_atoms=[1], axes=(0,), min_distance=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.npz")
            data.save_npz(path)
            loaded = AbInitioData.load_npz(path)
        np.testing.assert_allclose(loaded.points, data.points)
        np.testing.assert_allclose(loaded.energies, data.energies)
        np.testing.assert_allclose(loaded.gradients, data.gradients)
        self.assertEqual(loaded.provenance["backend"], "demo-lj")
        self.assertEqual(loaded.geometry.shape, data.geometry.shape)


class TestCLIDataPipeline(unittest.TestCase):
    """CLI sample → fit 闭环 (demo 后端, 无外部依赖)。"""

    def test_sample_then_fit(self):
        from aqpes.cli import main
        from aqpes.nn.model import PESNN

        with tempfile.TemporaryDirectory() as tmp:
            xyz = os.path.join(tmp, "ref.xyz")
            with open(xyz, "w") as f:
                f.write("2\nLJ dimer (Bohr)\nAr 0.0 0.0 0.0\nAr 3.4 0.0 0.0\n")
            npz = os.path.join(tmp, "d.npz")
            pkl = os.path.join(tmp, "m.pkl")

            import sys
            sys.argv = ["autoquantum", "sample", "--backend", "demo",
                        "--input", xyz, "-o", npz, "--n-per-dim", "5",
                        "--range", "-0.3", "0.3", "--active-atoms", "1",
                        "--min-distance", "1.5", "--max-points", "50"]
            self.assertEqual(main(), 0)
            self.assertTrue(os.path.exists(npz))

            data = AbInitioData.load_npz(npz)
            self.assertEqual(data.n_points, 50)

            sys.argv = ["autoquantum", "fit", "--data", npz, "-o", pkl,
                        "--epochs", "300", "--layers", "32", "32",
                        "--force-weight", "1.0"]
            self.assertEqual(main(), 0)
            model = PESNN.load(pkl)
            pred = model.predict(data.points)
            self.assertTrue(np.all(np.isfinite(pred)))
            # 力训练后拟合应达到有限精度 (演示级阈值)
            rmse = np.sqrt(np.mean((pred - data.energies) ** 2))
            span = data.energies.max() - data.energies.min()
            self.assertLess(rmse, 0.05 * span,
                            f"fit RMSE 过大: {rmse:.3e} (span {span:.3e})")


if __name__ == "__main__":
    unittest.main()


class TestCLISymmetryFit(unittest.TestCase):
    """CLI fit --symmetry: 置换不变委员会端到端。"""

    def test_symmetry_committee_cli(self):
        import sys
        import numpy as np
        from aqpes.cli import main
        from aqpes.nn.ensemble import AtomicEnergyCommittee

        rng = np.random.RandomState(0)
        n = 40
        base = np.array([[0.0, 0.0, 0.0], [1.43, 1.11, 0.0],
                         [-1.43, 1.11, 0.0]])
        c = np.repeat(base[None], n, axis=0) + rng.normal(0, 0.05,
                                                           (n, 3, 3))
        d = c[:, :, None, :] - c[:, None, :, :]
        r2 = np.maximum((d ** 2).sum(-1), 1e-6)
        iu = np.triu_indices(3, k=1)
        E = (0.35 / r2[:, iu[0], iu[1]]).sum(axis=1)

        with tempfile.TemporaryDirectory() as tmp:
            npz = os.path.join(tmp, "d.npz")
            pkl = os.path.join(tmp, "c.pkl")
            np.savez(npz, points=c.reshape(n, 9), energies=E,
                     symbols=np.array(["O", "H", "H"]))
            sys.argv = ["autoquantum", "fit", "--data", npz, "-o", pkl,
                       "--symmetry", "--committee", "2", "--epochs", "1200"]
            self.assertEqual(main(), 0)
            self.assertTrue(os.path.exists(pkl))
            com = AtomicEnergyCommittee.load(pkl)
            pred = com.predict(c)
            self.assertLess(float(np.sqrt(np.mean((pred - E) ** 2))),
                            0.6 * float(E.std()))
