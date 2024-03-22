from tinygrad import Tensor
from typing import Union


@staticmethod
def linspace(start: Union[int, float], stop: Union[int, float], steps: int, **kwargs):
    assert steps > -1, "number of steps must be non-negative"
    if steps == 0:
      return Tensor([], **kwargs)
    if steps == 1:
      return Tensor.full((1,), start, **kwargs)
    step_size = (stop - start) / (steps - 1)
    return Tensor.full((steps,), start, **kwargs) + Tensor.arange(steps, **kwargs) * step_size

class TriangularWindow:
    r"""Triangular window basis functions.

    Computes the basis functions

    .. math::
        \mathrm{triangular\_window}_k(x) = \max\left(
        \min\left(\frac{K}{l}x - k - 1, \frac{K}{l}x + k + 1\right),0\right)

    where :math:`k=0 \dots K-1` with :math:`K` = ``num`` and
    :math:`l` = ``limit``. Plot for :math:`K = 5` and :math:`l = 1`:

    .. jupyter-execute::
        :hide-code:

        import numpy as np, matplotlib.pyplot as plt
        import matplotlib_inline.backend_inline as inl
        from e3x.nn import triangular_window
        inl.set_matplotlib_formats('pdf', 'svg')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        x = np.linspace(0, 1.0, num=1001); K = 5; l = 1.0
        y = triangular_window(x, num=K, limit=l)
        plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{triangular\_window}_k(x)$')
        for k in range(K):
        plt.plot(x, y[:,k], lw=3, label=r'$k$'+f'={k}')
        plt.legend(); plt.grid()
        """
    def __init__(self, num: int, limit: int = 1.0):
        self.width = limit / num
        center = Tensor.linspace(0.0, limit, num=num + 1)[:-1]
        self.lower = center - self.width
        self.upper = center + self.width
  
    def __call__(self, x):
        x_1 = x.unsqueeze(-1)
        return Tensor.maximum(
            Tensor.minimum((x_1 - self.lower) / self.width, -(x_1 - self.upper) / self.width), 0
        )