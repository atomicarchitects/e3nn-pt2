r"""Functions related to irreducible representations of rotations."""

from typing import Optional, Tuple

import torch
from e3nn import io, o3

from .common import _so3_clebsch_gordan


class Irreps(o3.Irreps):

    # Just a couple of extra utilites over e3nn.Irreps
    @property
    def mul_dim(self):
        return max(mul for mul, irrep in self)

    @property
    def parity_dim(self):
        for _, irrep in self:
            if irrep.p == -1:
                return 2
        return 1

    def randn(
        self,
        leading_shape: Optional[Tuple[int, ...]] = (1,),
    ):
        r"""Random tensor

        Parameters
        ----------
        *leading_shape: batch shape which defaults to 1

        Returns
        ---------
        `torch.Tensor`
        tensor of shape ``(batch, parity, lmax, features)``

        """
        parity_dim_mapper = {-1: 2, 1: 1}
        padded_irreps_tensor = torch.zeros(
            leading_shape + (self.parity_dim, (self.lmax + 1) ** 2, self.mul_dim)
        )
        for _, irrep in self:
            parity_shape = parity_dim_mapper[irrep.p]
            padded_irreps_tensor[
                :, parity_shape - 1 : parity_shape, irrep.l**2 : (irrep.l + 1) ** 2, :
            ] = torch.randn(leading_shape + (1, 2 * irrep.l + 1, self.mul_dim))

        return padded_irreps_tensor


def clebsch_gordan(
    max_degree1: int,
    max_degree2: int,
    max_degree3: int,
    dtype: Optional[torch.dtype] = torch.float32,
):
    r"""Clebsch-Gordan coefficients for coupling all degrees at once.

    Args:
      max_degree1: Maximum degree of the first factor.
      max_degree2: Maximum degree of the second factor.
      max_degree3: Maximum degree of the tensor product.

    Returns:
      The values of all Clebsch-Gordan coefficients for coupling degrees up to the
      requested maximum degrees stored in an Array of shape
      ``((max_degree1+1)**2, (max_degree2+1)**2, (max_degree3+1)**2))``.
    """

    _l1_common = []
    for _l1 in range(max_degree1 + 1):
        _l2_common = []
        for _l2 in range(max_degree2 + 1):
            _l3_common = []
            for _l3 in range(max_degree3 + 1):
                _l3_common.append(_so3_clebsch_gordan(_l1, _l2, _l3))
            _l2_common.append(torch.cat(_l3_common, axis=2))
        _l1_common.append(torch.cat(_l2_common, axis=1))
    return torch.cat(_l1_common, axis=0).to(dtype)
