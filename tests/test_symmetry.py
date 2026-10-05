"""对称函数与共享原子能量委员会: 对称性物理正确性 + 委员会不确定性。"""

import os
import tempfile
import unittest
import numpy as np

from aqpes.nn.symmetry import (
    SymmetryFunctionSet, SymmetryFunctionParams)
from aqpes.nn.ensemble import (
    AtomicEnergyCommittee, AtomicTrainingConfig, train_atomic_committee)


def _sample_geometries(n=40, seed=0, symbols=("O", "H", "H")):
    """随机小振幅几何 (围绕参考构型) + 随机整体平移/旋转。"""
    rng = np.random.RandomState(seed)
    ref = np.array([[0.0, 0.0, 0.0],
                    [1.43, 1.11, 0.0],
                    [-1.43, 1.11, 0.0]])[:len(symbols)]
    c = np.repeat(ref[None], n, axis=0) + rng.normal(0, 0.06, (n, len(symbols), 3))
    c += rng.uniform(-3, 3, (n, 1, 3))              # 平移
    theta = rng.uniform(0, 2 * np.pi, n)             # z 轴旋转
    R = np.zeros((n, 3, 3))
    R[:, 0, 0] = np.cos(theta); R[:, 0, 1] = -np.sin(theta)
    R[:, 1, 0] = np.sin(theta); R[:, 1, 1] = np.cos(theta)
    R[:, 2, 2] = 1
    return np.einsum("nij,nkj->nki", R, c)


class TestSymmetryFunctions(unittest.TestCase):
    def setUp(self):
        self.symbols = ["O", "H", "H"]
        self.params = SymmetryFunctionParams(r_cut=4.0, radial_etas=(4.0, 10.0),
                                             angular_zetas=(1, 2))
        self.sf = SymmetryFunctionSet(self.symbols, self.params)
        self.coords = _sample_geometries(10, seed=1)

    def test_shape_and_finiteness(self):
        f = self.sf.compute(self.coords)
        self.assertEqual(f.shape, (10, 3, self.sf.n_features))
        self.assertTrue(np.all(np.isfinite(f)))

    def test_translation_rotation_invariance(self):
        f0 = self.sf.compute(self.coords)
        shifted = self.coords + np.array([1.3, -2.0, 0.7])
        np.testing.assert_allclose(self.sf.compute(shifted), f0, atol=1e-10)
        # 随机旋转
        rng = np.random.RandomState(3)
        A = rng.normal(size=(10, 3, 3))
        A /= np.linalg.norm(A, axis=(1, 2), keepdims=True)
        Q, _ = np.linalg.qr(A)
        rotated = np.einsum("nij,nkj->nki", Q, self.coords)
        np.testing.assert_allclose(self.sf.compute(rotated), f0, atol=1e-8)

    def test_permutation_equivariance(self):
        # 交换两个 H: 特征沿中心维置换 (总能量求和后不变)
        f0 = self.sf.compute(self.coords)
        perm = [0, 2, 1]
        symbols_p = [self.symbols[i] for i in perm]
        sf_p = SymmetryFunctionSet(symbols_p, self.params)
        f_p = sf_p.compute(self.coords[:, perm, :])
        # O 中心不变; H 中心互换 — 按符号对齐后逐中心一致
        np.testing.assert_allclose(f_p[:, 0], f0[:, 0], atol=1e-10)
        np.testing.assert_allclose(f_p[:, 1], f0[:, 2], atol=1e-10)
        np.testing.assert_allclose(f_p[:, 2], f0[:, 1], atol=1e-10)

    def test_cutoff_locality(self):
        # 移远一个 H 超出截断半径 → 其对 O 中心的径向贡献归零
        c = self.coords[:1].copy()
        c[0, 2, 0] = 20.0
        f_far = self.sf.compute(c)
        f_near = self.sf.compute(self.coords[:1])
        self.assertTrue(np.allclose(f_far[0, 0], f_near[0, 0]))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.sf.compute(np.zeros((5, 4, 3)))     # 原子数不符
        with self.assertRaises(ValueError):
            SymmetryFunctionSet(["H", "H"], SymmetryFunctionParams(r_cut=-1))


class TestAtomicCommittee(unittest.TestCase):
    def _toy_system(self, n=60, seed=0):
        """玩具 3 原子势: E = Σ_pairs A/r_ij² (严格置换+平移+旋转不变)。"""
        c = _sample_geometries(n, seed=seed)
        d = c[:, :, None, :] - c[:, None, :, :]
        r2 = np.maximum((d ** 2).sum(-1), 1e-6)
        iu = np.triu_indices(c.shape[1], k=1)
        E = (0.35 / r2[:, iu[0], iu[1]]).sum(axis=1)
        return c, E

    def test_committee_learns_invariant_energy(self):
        c, E = self._toy_system()
        committee, info = train_atomic_committee(
            ["O", "H", "H"], c, E, n_models=2,
            config=AtomicTrainingConfig(hidden_layers=(32, 32), epochs=600),
            sf_params=SymmetryFunctionParams(r_cut=4.0, radial_etas=(2.0, 5.0)),
            seed=0)
        self.assertLess(info["rmse"], 0.15 * np.std(E),
                        f"委员会拟合误差过大: {info['rmse']:.3e}")

    def test_energy_permutation_invariant(self):
        c, E = self._toy_system(n=40)
        committee, _ = train_atomic_committee(
            ["O", "H", "H"], c, E, n_models=2,
            config=AtomicTrainingConfig(hidden_layers=(24,), epochs=200),
            sf_params=SymmetryFunctionParams(r_cut=4.0, radial_etas=(3.0,),
                                             include_angular=False),
            seed=1)
        perm = [0, 2, 1]
        c_p = c[:, perm, :]
        committee_p = AtomicEnergyCommittee(
            [committee.symbols[i] for i in perm],
            committee.sf.params, committee.members)
        np.testing.assert_allclose(committee.predict(c_p),
                                   committee.predict(c), rtol=1e-10)

    def test_uncertainty_grows_out_of_distribution(self):
        c, E = self._toy_system(n=80)
        committee, _ = train_atomic_committee(
            ["O", "H", "H"], c, E, n_models=4,
            config=AtomicTrainingConfig(hidden_layers=(32, 32), epochs=300),
            sf_params=SymmetryFunctionParams(r_cut=4.0, radial_etas=(3.0,)),
            seed=2)
        # OOD: 把两个 H 拉远 (训练分布之外)
        c_ood = c[:10].copy()
        c_ood[:, 1:, 0] += 6.0
        u_in = committee.std(c[:10]).mean()
        u_ood = committee.std(c_ood).mean()
        self.assertGreater(u_ood, u_in,
                           f"OOD 不确定性未升高: {u_ood:.3e} vs {u_in:.3e}")

    def test_save_load_roundtrip(self):
        c, E = self._toy_system(n=30)
        committee, _ = train_atomic_committee(
            ["O", "H", "H"], c, E, n_models=2,
            config=AtomicTrainingConfig(hidden_layers=(16,), epochs=100),
            sf_params=SymmetryFunctionParams(r_cut=4.0, radial_etas=(3.0,),
                                             include_angular=False),
            seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "com.pkl")
            committee.save(path)
            loaded = AtomicEnergyCommittee.load(path)
        np.testing.assert_allclose(loaded.predict(c), committee.predict(c),
                                   rtol=1e-12)


if __name__ == "__main__":
    unittest.main()


class TestActiveLearning(unittest.TestCase):
    """主动学习闭环: 分歧选点 vs 随机选点。"""

    def _toy(self, c):
        c = np.asarray(c)
        squeeze = c.ndim == 2
        if squeeze:
            c = c[None]
        d = c[:, :, None, :] - c[:, None, :, :]
        r2 = np.maximum((d ** 2).sum(-1), 1e-6)
        iu = np.triu_indices(c.shape[1], k=1)
        E = (0.35 / r2[:, iu[0], iu[1]]).sum(axis=1)
        return E[0] if squeeze else E

    def test_loop_reduces_pool_rmse(self):
        from aqpes.nn.active_learning import run_active_learning
        from aqpes.nn.ensemble import AtomicTrainingConfig
        from aqpes.pes.calculators import AnalyticCalculator

        symbols = ["O", "H", "H"]
        rng = np.random.RandomState(5)

        def toy(c):
            c = np.asarray(c)
            sq = c.ndim == 2
            if sq:
                c = c[None]
            d = c[:, :, None, :] - c[:, None, :, :]
            r2 = np.maximum((d ** 2).sum(-1), 1e-6)
            iu = np.triu_indices(c.shape[1], k=1)
            E = (0.35 / r2[:, iu[0], iu[1]]).sum(axis=1)
            return E[0] if sq else E

        # 训练域 (小振幅) 与候选池 (更大振幅, 委员会最初外推)
        c0 = _sample_geometries(16, seed=7)
        pool = _sample_geometries(50, seed=8)
        pool += rng.normal(0, 0.10, pool.shape)   # 更偏离初始分布
        calc = AnalyticCalculator(toy)

        result = run_active_learning(
            symbols, calc, c0, pool, n_iterations=3, batch_size=10,
            n_models=2,
            training_config=AtomicTrainingConfig(hidden_layers=(24, 24),
                                                 epochs=300),
            sf_params=SymmetryFunctionParams(r_cut=4.0, radial_etas=(2., 5.)),
            seed=0, verbose=False)

        self.assertEqual(len(result.history), 3)
        self.assertLess(result.history[-1]["n_labeled"],
                        16 + 3 * 10 + 1)              # 无重复挑点
        # 主动学习后池 RMSE 显著低于首轮 (初始标注少 + 外推区)
        self.assertLess(result.pool_rmse_history[-1],
                        0.7 * result.pool_rmse_history[0],
                        f"池 RMSE 未下降: {result.pool_rmse_history}")
        # 挑过的点不再被挑
        picked = [j for h in result.history for j in h["picked_indices"]]
        self.assertEqual(len(picked), len(set(picked)))

    def test_loop_stops_when_pool_exhausted(self):
        from aqpes.nn.active_learning import run_active_learning
        from aqpes.nn.ensemble import AtomicTrainingConfig
        from aqpes.pes.calculators import AnalyticCalculator

        c0 = _sample_geometries(10, seed=1)
        pool = _sample_geometries(6, seed=2)
        calc = AnalyticCalculator(self._toy)
        result = run_active_learning(
            ["O", "H", "H"], calc, c0, pool, n_iterations=5, batch_size=10,
            n_models=2,
            training_config=AtomicTrainingConfig(hidden_layers=(16,),
                                                 epochs=100),
            sf_params=SymmetryFunctionParams(r_cut=4.0, radial_etas=(2., 5.)),
            seed=1, verbose=False)
        self.assertLessEqual(len(result.history), 5)
        self.assertEqual(len(result.labeled_coords),
                         10 + 6)   # 池全部标注完即停


if __name__ == "__main__":
    unittest.main()
