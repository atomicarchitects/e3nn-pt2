import torch

from e3nn import io, o3
from e3nn_pt2 import so3

from typing import Tuple, List, Any, Sequence, Union, Optional
import jaxtyping

torch.set_float32_matmul_precision('high')


class TensorProduct(torch.nn.Module):

    def __init__(self, irreps_in1, irreps_in2, device="cpu"):
        super().__init__()
        self.irreps_out = so3.Irreps(o3.FullTensorProduct(irreps_in1, irreps_in2).irreps_out.__str__())
        self.cg = so3.clebsch_gordan(irreps_in1.lmax, irreps_in2.lmax, self.irreps_out.lmax).to(device=device)
        self.pseudo_tensor_in1 = irreps_in1.parity_dim == 2
        self.pseudo_tensor_in2 = irreps_in2.parity_dim == 2
        
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
        if (not self.pseudo_tensor_in1) and (not self.pseudo_tensor_in2):
            return torch.einsum(f'...lf, ...m{self.channel_dim}, lmn-> ...nf', x1, x2, self.cg)

        def _couple_slices(i: int, j: int):
            return torch.einsum(
                f'...lf, ...m{self.channel_dim}, lmn -> ...nf',
                x1[..., i, :, :],
                x2[..., j, :, :],
                self.cg,
            )
        
        even_parity_output = odd_parity_output = None
        if (not self.pseudo_tensor_in1) and (not self.pseudo_tensor_in2):
            eee = _couple_slices(0, 0) # even + even -> even
            even_parity_output = eee
        if (self.pseudo_tensor_in1) and (not self.pseudo_tensor_in2):
            oeo = _couple_slices(1, 0)  # odd + even -> odd
            odd_parity_output = oeo
        if (self.pseudo_tensor_in1) and (self.pseudo_tensor_in2):
            ooe = _couple_slices(1, 1)  # odd + odd -> even
            even_parity_output += ooe
        if (not self.pseudo_tensor_in1) and (self.pseudo_tensor_in2):
            eoo = _couple_slices(0, 1)  # even + odd -> odd
            odd_parity_output += eoo

        # Combine same parities and return stacked features.
        return torch.stack((even_parity_output, odd_parity_output), axis=-3)
