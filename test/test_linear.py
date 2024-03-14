import unittest
import tinye3nn


class TestLinear(unittest.TestCase):
    def test_linear_fwd_e3(self):
        x_irreps = tinye3nn.so3.Irreps("32x0e + 32x1e + 32x2e")
        y_irreps = tinye3nn.so3.Irreps("0e + 1e + 2e")
        x = x_irreps.randn()
        mod = tinye3nn.nn.Linear(x_irreps, y_irreps)
        out = mod(x)

    def test_linear_fwd_se3(self):
        x_irreps = tinye3nn.so3.Irreps("32x0e + 32x1o + 32x2e")
        y_irreps = tinye3nn.so3.Irreps("0e + 1o + 2e")
        x = x_irreps.randn()
        mod = tinye3nn.nn.Linear(x_irreps, y_irreps)
        out = mod(x)


if __name__ == "__main__":
    unittest.main()
