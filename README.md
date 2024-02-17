# e3nn-pt2

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

Your friendly neighborhood [e3nn](https://github.com/e3nn/e3nn/) powered by a [e3x](https://github.com/google-research/e3x/) memory backend with first class support for [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)

## Installation
```console
> python -m pip install .
```

## Usage

```python
import e3nn_pt2

# Create Irreps as per e3nn convention (currently not supporting other input formats except strings)

x_irreps = e3nn_pt2.so3.Irreps("32x0e + 32x1e + 32x2e")
y_irreps = e3nn_pt2.so3.Irreps("0e + 1o + 2e")


# Initialize arrays and move them to the GPU

x = x_irreps.randn().to(device='cuda')
y = y_irreps.randn().to(device='cuda')

# Initialize TP + Linear module with the option to specify the device
mod = e3nn_pt2.nn.TensorProductLinear(x_irreps, y_irreps, device='cuda')

# Run the model :)

mod(x_irreps, y_irreps)

```
## Acknowledgement

- e3nn
- e3nn-jax
- e3x
