from e3nn import o3
from e3nn_pt2 import so3

from typing import Tuple, List, Any, Sequence, Union, Optional
from tinygrad import Tensor, nn
# Linear layer

class Linear:
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

    def __init__(
        self,
        irreps_in: so3.Irreps,
        irreps_out: so3.Irreps,
        use_bias: Optional[int] = False,
    ):
        self.pseudo_tensor = (irreps_in.parity_dim == 2) or (irreps_out.parity_dim == 2)
        self.lmax = irreps_in.lmax
        self.output_mul = irreps_out.mul_dim
        self.ells = [irrep.l for _, irrep in irreps_in]
        for lmax in self.ells:
            if self.pseudo_tensor:  # Has pseudotensors.
                setattr(self, f"dense_e_{lmax}",
                    nn.Linear(in_features=irreps_in.mul_dim,
                              out_features=irreps_out.mul_dim,
                              bias=use_bias))
                setattr(self, f"dense_o_{lmax}",
                    nn.Linear(in_features=irreps_in.mul_dim,
                              out_features=irreps_out.mul_dim,
                              bias=False))

            else:
                setattr(self, f"dense_{lmax}",
                    nn.Linear(in_features=irreps_in.mul_dim,
                              out_features=irreps_out.mul_dim,
                              bias=use_bias))
    def __call__(
        self,
        inputs):
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """

        if self.pseudo_tensor:
            return Tensor.stack(
                [
                    # Even parity (tensors).
                    Tensor.cat(
                        *[
                            (
                                getattr(self, f"dense_e_{lmax}")(
                                    inputs[..., 0, lmax**2 : (lmax + 1) ** 2, :]
                                )
                                if lmax in self.ells
                                else inputs[..., 0, lmax**2 : (l + 1) ** 2, :].expand(
                                    inputs.shape[:-1] + (self.output_mul,)
                                )
                            )
                            for lmax in range(self.lmax + 1)
                        ],
                        dim=-2,
                    ),
                    # Odd parity (pseudotensors).
                    Tensor.cat(
                        *[
                            (
                                getattr(self, f"dense_o_{lmax}")(
                                    inputs[..., 1, lmax**2 : (lmax + 1) ** 2, :]
                                )
                                if lmax in self.ells
                                else inputs[..., 1, lmax**2 : (lmax + 1) ** 2, :].expand(
                                    inputs.shape[:-1] + (self.output_mul,)
                                )
                            )
                            for lmax in range(self.lmax + 1)
                        ],
                        dim=-2,
                    ),
                ],
                dim=-3,
            )
        else:  # Has no pseudotensors.
            return Tensor.cat(
                *[
                    (
                        getattr(self, f"dense_{lmax}")(inputs[..., lmax**2 : (lmax + 1) ** 2, :])
                        if lmax in self.ells
                        else inputs[..., lmax**2 : (lmax + 1) ** 2, :].expand(
                            inputs.shape[:-1] + (self.output_mul,)
                        )
                    )
                    for lmax in range(self.lmax + 1)
                ],
                dim=-2,
            )