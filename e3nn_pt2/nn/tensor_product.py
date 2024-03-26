import torch
from torch import nn
import numpy as np

from e3nn import io, o3
from e3nn_pt2 import so3

from typing import Tuple, List, Any, Sequence, Union, Optional
import jaxtyping

torch.set_float32_matmul_precision("high")

class TensorProduct(nn.Module):
    def __init__(self, irreps_in1, irreps_in2, batch=1):
        super().__init__()
        self.irreps_out = so3.Irreps(
            o3.FullTensorProduct(irreps_in1, irreps_in2).irreps_out.__str__()
        )
        self.cg = nn.Parameter(
            so3.clebsch_gordan(irreps_in1.lmax, irreps_in2.lmax, self.irreps_out.lmax)
        )
        self.pseudo_tensor_in1 = irreps_in1.parity_dim == 2
        self.pseudo_tensor_in2 = irreps_in2.parity_dim == 2
        self.parity_masks = nn.ParameterDict(
            {
                "even": torch.Tensor([1, 0]).reshape(2, 1),
                "first_dim": torch.Tensor([[1, 0], [0, 1], [1, 0], [0, 1]]),
                "second_dim": torch.Tensor([[1, 0], [0, 1], [0, 1], [1, 0]]),
                "concatenate": torch.Tensor([[1, 1, 0, 0], [0, 0, 1, 1]]),
            }
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

    def forward(self, x1, x2):
        if (not self.pseudo_tensor_in1) and (not self.pseudo_tensor_in2):
            return torch.einsum(
                f"bplf, bpm{self.channel_dim}, ap, lmn -> banf",
                x1,
                x2,
                self.parity_masks["even"],
                self.cg,
            )

        else:
            return torch.einsum(
                f"bplf, ap, bql{self.channel_dim}, aq, lmn, za -> bznf",
                x1,
                self.parity_masks["first_dim"],
                x2,
                self.parity_masks["second_dim"],
                self.cg,
                self.parity_masks["concatenate"],
            )
