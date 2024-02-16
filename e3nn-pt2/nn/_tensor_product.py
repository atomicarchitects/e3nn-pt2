import torch

from e3nn import io, o3
from e3x_torch import so3

from typing import Tuple, List, Any, Sequence, Union, Optional
import jaxtyping

torch.set_float32_matmul_precision('high')


class TensorProduct(torch.nn.Module):

    def __init__(self, irreps_in1, irreps_in2, device="cpu"):
        super().__init__()
        self.irreps_out = so3.Irreps(o3.FullTensorProduct(irreps_in1, irreps_in2).irreps_out.__str__())
        self.cg = so3.clebsch_gordan(irreps_in1.lmax, irreps_in2.lmax, self.irreps_out.lmax).to(device=device)
        self.pseudo_tensor = ((irreps_in1.parity_dim == 2) or (irreps_in2.parity_dim == 2))
        
        # Check if number of channels is same for the 2 inputs
        # else use the channel axis from the first input
        # typically useful for spherical type opertions
        # e.g 128x0e + 128x1o ... x 1x0e + 1x1o -> 128x0e + 128x1o ... 

        if irreps_in1.mul_dim == irreps_in2.mul_dim :
            self.channel_dim = 'f' # f x f -> f
        else:
            self.channel_dim = 'g' # f x g -> f 
        
    @torch.compile(fullgraph=True)
    def forward(self, x1, x2):
        if not self.pseudo_tensor:
            return torch.einsum(f'...lf, ...m{self.channel_dim}, lmn->...nf', x1, x2, self.cg)

        def _couple_slices(i: int, j: int):
            return torch.einsum(
                f'...lf, ...m{self.channel_dim}, lmn -> ...nf',
                x1[..., i, :, :],
                x2[..., j, :, :],
                self.cg,
            )

        eee = _couple_slices(0, 0) # even + even -> even
        ooe = _couple_slices(1, 1)  # odd + odd -> even
        eoo = _couple_slices(0, 1)  # even + odd -> odd
        oeo = _couple_slices(1, 0)  # odd + even -> odd

        # Combine same parities and return stacked features.
        return torch.stack((eee + ooe, eoo + oeo), axis=-3)