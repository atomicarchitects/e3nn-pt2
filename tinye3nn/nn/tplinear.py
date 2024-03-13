from tinye3nn import so3, nn
from e3nn import o3
from typing import Optional


class TensorProductLinear:
    def __init__(
        self,
        irreps_in1: so3.Irreps,
        irreps_in2: so3.Irreps,
        batch: int = 1,
        use_bias: Optional[int] = False,
    ):
        super().__init__()
        self.tp = nn.TensorProduct(irreps_in1, irreps_in2, batch)
        # Assuming that the first input has the channels
        self.linear = nn.Linear(irreps_in=irreps_in1, irreps_out=irreps_in1)

    def __call__(self, x_irreps, y_irreps):
        output = self.tp(x_irreps, y_irreps)
        output = self.linear(output)
        return output
