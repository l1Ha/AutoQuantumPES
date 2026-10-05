import unittest
import numpy as np
from aqpes.pes.leps import LEPSPES, LEPSBuilder


class TestLEPS(unittest.TestCase):
    def setUp(self):
        self.params = {
            "D_HH": 0.1744, "alpha_HH": 1.028, "r0_HH": 1.401,
            "sato": 0.15,
        }

    def test_leps_evaluate(self):
        pes = LEPSPES(self.params)
        v = pes.evaluate(np.array([1.5]), np.array([1.5]), np.array([3.0]))
        self.assertEqual(v.shape, (1,))
        self.assertFalse(np.isnan(v[0]))

    def test_leps_builder_grid(self):
        builder = LEPSBuilder(self.params)
        R, r, V = builder.generate_grid((0.5, 5.0), (0.5, 5.0), 50, 50)
        self.assertEqual(V.shape, (50, 50))
        self.assertFalse(np.any(np.isnan(V)))

    def test_leps_2d_eval(self):
        builder = LEPSBuilder(self.params)
        v = builder.evaluate_2d(np.array([2.0]), np.array([1.4]))
        self.assertFalse(np.isnan(v))


if __name__ == "__main__":
    unittest.main()
