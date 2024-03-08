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
                "odd": torch.Tensor([0, 1]).reshape(2, 1),
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

    @torch.compile
    def forward(self, x1, x2):
        if (not self.pseudo_tensor_in1) and (not self.pseudo_tensor_in2):
            return torch.einsum(
                f"...lf, ...m{self.channel_dim}, lmn -> ...nf",
                torch.einsum("bplf, pa -> balf", x1, self.parity_masks["even"]),
                torch.einsum("bplf, pa -> balf", x2, self.parity_masks["even"]),
                self.cg,
            )

        else:

            def _couple_slices(x1_mask_idx: int, x2_mask_idx: int):
                return torch.einsum(
                    f"...lf, ...m{self.channel_dim}, lmn -> ...nf",
                    torch.einsum(
                        "bplf, pa -> balf", x1, self.parity_masks[x1_mask_idx]
                    ),
                    torch.einsum(
                        "bplf, pa -> balf", x2, self.parity_masks[x2_mask_idx]
                    ),
                    self.cg,
                )

            eee = _couple_slices("even", "odd")  # even + even -> even
            ooe = _couple_slices("odd", "even")  # odd + odd -> even
            eoo = _couple_slices("even", "odd")  # even + odd -> odd
            oeo = _couple_slices("odd", "odd")  # odd + even -> odd

        # Combine same parities and return stacked features.
        return torch.stack((eee + ooe, eoo + oeo), axis=-3)
