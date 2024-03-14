import unittest
import tinye3nn


class TestTensorProduct(unittest.TestCase):
    def test_tplinear_fwd_e3(self):
        x_irreps = tinye3nn.so3.Irreps("32x0e + 32x1e + 32x2e")
        y_irreps = tinye3nn.so3.Irreps("0e + 1e + 2e")
        x = x_irreps.randn()
        y = y_irreps.randn()
        mod = tinye3nn.nn.TensorProductLinear(x_irreps, y_irreps)
        out = mod(x, y)

    def test_tplinear_fwd_se3(self):
        x_irreps = tinye3nn.so3.Irreps("32x0e + 32x1o + 32x2e")
        y_irreps = tinye3nn.so3.Irreps("0e + 1o + 2e")
        x = x_irreps.randn()
        y = y_irreps.randn()
        mod = tinye3nn.nn.TensorProductLinear(x_irreps, y_irreps)
        out = mod(x, y)


if __name__ == "__main__":
    unittest.main()
