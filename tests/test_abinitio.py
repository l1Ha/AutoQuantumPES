import unittest
import numpy as np

from aqpes.pes.abinitio import AbInitioData


class TestSampleFunction(unittest.TestCase):
    def test_gradients_match_analytic_1d(self):
        # V(x) = 0.5·k·x² → dV/dx = k·x
        k = 0.7
        data = AbInitioData.sample_function(
            lambda pts: 0.5 * k * pts[:, 0] ** 2,
            ranges=[(-2.0, 2.0)], n_per_dim=41)
        np.testing.assert_allclose(data.gradients[:, 0], k * data.points[:, 0],
                                   rtol=1e-4, atol=1e-6)
        self.assertEqual(data.points.shape, (41, 1))

    def test_gradients_match_analytic_2d(self):
        # V(x, y) = sin(x)·exp(-y²/2) → dV/dx = cos(x)e^{-y²/2}, dV/dy = -y·V
        f = lambda pts: (np.sin(pts[:, 0])
                         * np.exp(-0.5 * pts[:, 1] ** 2))
        data = AbInitioData.sample_function(f, ranges=[(-2, 2), (-2, 2)],
                                            n_per_dim=31)
        x, y = data.points[:, 0], data.points[:, 1]
        v = f(data.points)
        np.testing.assert_allclose(data.gradients[:, 0], np.cos(x) * np.exp(-0.5 * y ** 2),
                                   rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(data.gradients[:, 1], -y * v,
                                   rtol=1e-4, atol=1e-6)

    def test_split_keeps_gradients_aligned(self):
        data = AbInitioData.sample_function(
            lambda pts: pts[:, 0] ** 2, ranges=[(0, 1)], n_per_dim=20)
        train, test = data.split(0.7)
        self.assertGreater(train.n_points, 0)
        self.assertIsNotNone(train.gradients)
        # 对齐性: 梯度 = 2x
        np.testing.assert_allclose(train.gradients[:, 0], 2.0 * train.points[:, 0],
                                   rtol=1e-4, atol=1e-6)

    def test_from_arrays_1d_input(self):
        data = AbInitioData()
        data.from_arrays(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        self.assertEqual(data.points.shape, (3, 1))
        self.assertEqual(data.n_points, 3)


if __name__ == "__main__":
    unittest.main()
