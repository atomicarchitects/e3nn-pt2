import torch


from e3nn import io, o3
from e3nn_pt2 import so3

from typing import Tuple, List, Any, Sequence, Union, Optional
import jaxtyping
torch.set_float32_matmul_precision('high')

Float = jaxtyping.Float
Array = torch.Tensor

# Linear layer

def _make_dense_for_each_degree(
    ells: List[int],
    in_features: int,
    out_features: int,
    use_bias: bool, 
    device: str
    ) -> List[torch.nn.Linear]:
    """Helper function for generating Modules."""
    dense = torch.nn.ModuleDict({})
    for l in ells:
        dense[f"{l}"] = torch.nn.Linear(
                            in_features=in_features,
                            out_features=out_features,
                            bias=use_bias).to(device=device)
    return dense

class Linear(torch.nn.Module):
    r"""A linear transformation applied over the last dimension of the input.

    The transformation can be written as

    .. math::

        \mathbf{y}^{(\ell_p)} = \begin{cases}
        \mathbf{x}^{(\ell_p)}\mathbf{W}_{(\ell_p)} + \mathbf{b} & \ell_p = 0_+ \\
        \mathbf{x}^{(\ell_p)}\mathbf{W}_{(\ell_p)} & \ell_p \neq 0_+
        \end{cases}

    where
    :math:`\mathbf{x} \in \mathbb{R}^{P\times (L+1)^2 \times F_{\mathrm{in}}}` and
    :math:`\mathbf{y} \in \mathbb{R}^{P\times (L+1)^2 \times F_{\mathrm{out}}}`
    are the inputs and outputs, respectively. Here, :math:`P` is either :math:`1`
    or :math:`2` (depending on whether the inputs contain pseudotensors or not),
    :math:`L` is the maximum degree of the input features, and
    :math:`F_{\mathrm{in}}` and :math:`F_{\mathrm{out}}` = ``features`` are the
    number of input and output features. Every combination of degree :math:`\ell`
    and parity :math:`p` has separate weight matrices
    :math:`\mathbf{W}_{(\ell_p)}`. Note that a bias term
    :math:`\mathbf{b} \in \mathbb{R}^{1\times 1 \times F_{\mathrm{out}}}` is only
    applied to the scalar channel (:math:`\ell_p= 0_+`) when ``use_bias=True``.

    Attributes:
        irreps_in: Irreps
        irreps_out: Irreps
        use_bias: Whether to add a bias to the scalar channel of the output.
    """

    def __init__(self,
                irreps_in: so3.Irreps,
                irreps_out: so3.Irreps,
                device: str = "cpu",
                use_bias: Optional[int] = False
                ):
        super().__init__()
        self.pseudo_tensor = (irreps_in.parity_dim == 2) or (irreps_out.parity_dim == 2)
        self.lmax = irreps_in.lmax
        self.output_mul = irreps_out.mul_dim
        self.ells = [irrep.l for _, irrep in irreps_in]
        self.layers = {}
        if self.pseudo_tensor:  # Has pseudotensors.
            self.dense_e = _make_dense_for_each_degree(self.ells, irreps_in.mul_dim, irreps_out.mul_dim, use_bias, device, '+')
            self.dense_o = _make_dense_for_each_degree(self.ells, irreps_in.mul_dim, irreps_out.mul_dim, False, device, '-')
        
        else:
            self.dense = _make_dense_for_each_degree(self.ells, irreps_in.mul_dim, irreps_out.mul_dim, use_bias, device)
    
    
    @torch.compile(fullgraph=True)
    def forward(
        self,
        inputs: Union[
            Float[Array, '... 1 (max_degree+1)**2 in_features'],
            Float[Array, '... 2 (max_degree+1)**2 in_features'],
        ],
    ) -> Union[
        Float[Array, '... 1 (max_degree+1)**2 out_features'],
        Float[Array, '... 2 (max_degree+1)**2 out_features'],
    ]:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
    
        if self.pseudo_tensor:
            return torch.stack(
                [
                    # Even parity (tensors).
                    torch.cat(
                        [
                            self.dense_e[f"{l}"](inputs[..., 0, l**2 : (l + 1) ** 2, :]) if l in self.ells
                            else inputs[..., 0, l**2 : (l + 1) ** 2, :].expand(inputs.shape[:-1]+(self.output_mul,))
                            for l in range(self.lmax+1)
                        ],
                        axis=-2,
                    ),
                    # Odd parity (pseudotensors).
                    torch.cat(
                        [
                            self.dense_o[f"{l}"](inputs[..., 1, l**2 : (l + 1) ** 2, :]) if l in self.ells
                            else inputs[..., 1, l**2 : (l + 1) ** 2, :].expand(inputs.shape[:-1]+(self.output_mul,))
                            for l in range(self.lmax+1)
                        ],
                        axis=-2,
                    ),
                ],
                axis=-3,
            )
        else:  # Has no pseudotensors.
            return torch.cat(
                [
                    self.dense[f"{l}"](inputs[..., l**2 : (l + 1) ** 2, :]) if l in self.ells
                    else inputs[..., l**2 : (l + 1) ** 2, :].expand(inputs.shape[:-1]+(self.output_mul,))
                    for l in range(self.lmax+1)
                ],
                axis=-2,
            )
