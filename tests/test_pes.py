import unittest
import numpy as np
from aqpes.pes.analytic import MorsePES, HarmonicPES, LJ_PES


class TestPESAnalytic(unittest.TestCase):
    def test_morse_shape(self):
        params = {"D": 0.1744, "alpha": 1.028, "r0": 0.7416}
        pes = MorsePES(params)
        x = np.linspace(0.3, 5.0, 100)
        v = pes(x)
        self.assertEqual(v.shape, x.shape)
        self.assertAlmostEqual(v.min(), 0.0, places=2)

    def test_harmonic(self):
        pes = HarmonicPES({"k": 1.0, "r0": 0.7416})
        self.assertAlmostEqual(pes.evaluate(np.array([0.7416]))[0], 0.0, places=6)

    def test_lj(self):
        pes = LJ_PES({"epsilon": 1.0, "sigma": 1.0})
        v = pes(np.array([1.0]))
        self.assertAlmostEqual(v[0], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
