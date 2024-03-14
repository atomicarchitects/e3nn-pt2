import numpy as np
from tinygrad import Tensor, Device, dtypes

from e3nn import o3
from tinye3nn import so3

from typing import Tuple, List, Any, Sequence, Union, Optional


class TensorProduct:
    def __init__(self, irreps_in1, irreps_in2, batch=1):
        self.cg = so3.clebsch_gordan(
            irreps_in1.lmax, irreps_in2.lmax, irreps_in1.lmax + irreps_in2.lmax
        )
        self.pseudo_tensor_in1 = irreps_in1.parity_dim == 2
        self.pseudo_tensor_in2 = irreps_in2.parity_dim == 2
        self.even = Tensor([1, 0]).reshape(2, 1)
        self.first_dim = Tensor(
            [[1, 0], [0, 1], [1, 0], [0, 1]], requires_grad=True, device=Device.DEFAULT
        )
        self.second_dim = Tensor(
            [[1, 0], [0, 1], [0, 1], [1, 0]], requires_grad=True, device=Device.DEFAULT
        )
        self.concatenate = Tensor(
            [[1, 1, 0, 0], [0, 0, 1, 1]], requires_grad=True, device=Device.DEFAULT
        )

        # TODO: Add named tensors
        # TODO Simplify this logic

        # Check if number of channels is same for the 2 inputs
        # else use the channel axis from the first input
        # typically useful for spherical type opertions
        # e.g 128x0e + 128x1o ... x 1x0e + 1x1o -> 128x0e + 128x1o ...

        if irreps_in1.mul_dim == irreps_in2.mul_dim:
            self.channel_dim = "f"  # f x f -> f
        else:
            self.channel_dim = "g"  # f x g -> f

    def __call__(self, x1, x2):
        if (not self.pseudo_tensor_in1) and (not self.pseudo_tensor_in2):
            return Tensor.einsum(
                f"bplf, bpm{self.channel_dim}, ap, lmn -> banf",
                x1,
                x2,
                self.even,
                self.cg,
            )

        else:
            return Tensor.einsum(
                f"bplf, ap, bql{self.channel_dim}, aq, lmn, za -> bznf",
                x1,
                self.first_dim,
                x2,
                self.second_dim,
                self.cg,
                self.concatenate,
            )
