from typing import Callable, Optional
import functools
from tinygrad import Tensor
import tinye3nn

class BasisWrapper:
    r"""Convenience wrapper for computing radial-angular basis functions."""

    def __init__(
        self,
        max_degree: int,
        radial_fn: Callable,
        angular_fn = tinye3nn.so3.SphericalHarmonics,
        cartesian_order: bool = False,):

        self.rbf = radial_fn
        self.ylm = angular_fn(max_degree=max_degree, cartesian_order=cartesian_order)
    
    def __call__(self, r):
        # Check that r is a collection of 3-vectors.
        if r.shape[-1] != 3:
            raise ValueError(f'r must have shape (..., 3), received shape {r.shape}')

        # Normalize input vectors.
        norm = tinye3nn.ops.norm(r, axis=-1, keepdim=True)  # (..., 1)
        u = r / Tensor.where(norm > 0, norm, 1)
        norm = norm.squeeze(-1)  # (...)

        # Evaluate radial basis functions.
        rbf = self.rbf(norm)  # (..., N)

        # Evaluate angular basis functions.
        ylm = self.ylm(r=u)

        # Combine radial and angular basis functions.
        ylm = ylm.unsqueeze(-1)  # (..., (L+1)**2, 1)
        rbf = rbf.unsqueeze(-2)  # (..., 1, N)
        out = ylm * rbf  # (..., (L+1)**2, N)

        # Add parity axis.
        out = out.unsqueeze(-3)  # (..., 1, (L+1)**2, N)

        return out