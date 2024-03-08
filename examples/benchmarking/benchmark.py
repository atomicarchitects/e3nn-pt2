import argparse

parser = argparse.ArgumentParser(description="Script with mode flag")
parser.add_argument(
    "--mode",
    type=str,
    choices=["torch", "jax"],
    default="jax",
    help="Select mode: 'torch' or 'jax'",
)
parser.add_argument("--batch", type=int, default=1, help="Select batch size")
parser.add_argument(
    "--features", type=int, default=128, help="Number of feature channels"
)
parser.add_argument("--lmax", type=int, default=7, help="lmax for the operation")
parser.add_argument(
    "--runs", type=int, default=1000, help="Number of timing runs for benchmarking"
)
parser.add_argument(
    "--warmup", type=int, default=100, help="Number of warmup runs for benchmarking"
)
parser.add_argument(
    "--all_even",
    type=bool,
    default=False,
    help="Choose whether irreps are all even or not",
)
parser.add_argument(
    "--plot", type=bool, default=True, help="Choose whether to make plots or not"
)

arg_parser = parser.parse_args()

if arg_parser.mode == "torch":
    import torch
    from e3nn import io, o3, util
    import e3nn_pt2
    import functools

    def TPLinear_E3NNPT2(tp_e3nnpt2, x, y):
        # def loss_function(params, inputs):
        #     return torch.sum(torch.tanh(
        #                 torch.func.functional_call(
        #                     mod,
        #                     dict(mod.named_parameters()),
        #                     (x, y,))))
        # return torch.func.grad_and_value(loss_function, argnums=0)(x, y)
        out = tp_e3nnpt2(x, y)
        return out, out.tanh().sum().backward()

    def TPLinear_E3NNPT(tp_e3nn, x_e3nn, y_e3nn):
        out = tp_e3nn(x_e3nn, y_e3nn)
        return out, out.tanh().sum().backward()


if arg_parser.mode == "jax":
    import os

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
    )
    from flax import linen as nn
    import jax
    import jax.numpy as jnp
    import e3x
    import e3nn_jax

    import functools

    class TPLineare3x(nn.Module):
        max_degree: int
        features: int

        @nn.compact
        def __call__(self, x_irreps, y_irreps):
            return e3x.nn.Dense(features=self.features)(
                e3x.nn.Tensor(max_degree=self.max_degree, include_pseudotensors=True)(
                    x_irreps, y_irreps
                )
            )

    @jax.jit
    def TPLinear_E3X(kernel, x, y):
        def loss_function(kernel, x, y):
            return jnp.sum(jnp.tanh(tplinear_e3x.apply(kernel, x, y)))

        return jax.value_and_grad(loss_function)(kernel, x, y)

    class TPLineare3nn(nn.Module):
        irreps_out: e3nn_jax.Irreps

        @nn.compact
        def __call__(self, x_irreps_e3nn, y_irreps_e3nn):
            return e3nn_jax.flax.Linear(self.irreps_out)(
                e3nn_jax.tensor_product(x_irreps_e3nn, y_irreps_e3nn)
            )

    @jax.jit
    def TPLinear_E3NN(kernel, x_irreps_e3nn, y_irreps_e3nn):
        def loss_function(kernel, x_irreps_e3nn, y_irreps_e3nn):
            return sum(
                jnp.sum(jnp.tanh(out))
                for out in jax.tree_util.tree_leaves(
                    tplinear_e3nn.apply(kernel, x_irreps_e3nn, y_irreps_e3nn)
                )
            )

        return jax.value_and_grad(loss_function)(kernel, x_irreps_e3nn, y_irreps_e3nn)


from time import time
import numpy as np


def run(func, *args, mode="torch"):
    print("Warmup....")
    for _ in range(arg_parser.warmup):
        result = func(*args)

    print("Benchmarking")
    start = time()
    for _ in range(arg_parser.runs):
        result = func(*args)
        if mode == "jax":
            result = func(*args)
            # Check for IrrepsArray
            if hasattr(result, "array"):
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result.array)
            else:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
        else:
            torch.cuda.synchronize()
    end = time()

    return end - start


def byte_counter(x, mode="torch"):
    if mode == "torch":
        num_elements = x.numel()
        element_size = x.element_size()
        return num_elements * element_size
    else:
        num_elements = jnp.size(x)
        element_size = jnp.dtype(x.dtype).itemsize
        return num_elements * element_size


e3nn_jax_times = []
e3x_times = []
e3x_torch_times = []
e3nn_torch_times = []

for lmax in range(1, arg_parser.lmax + 1):
    if arg_parser.mode == "torch":
        irreps_in1 = o3.Irreps(
            [
                (arg_parser.features, (l, (-1) ** (0 if arg_parser.all_even else l)))
                for l in range(lmax + 1)
            ]
        )
        irreps_in1_pt2 = e3nn_pt2.so3.Irreps(irreps_in1)
        print(irreps_in1)
        irreps_in2 = o3.Irreps(
            [
                (1, (l, (-1) ** (0 if arg_parser.all_even else l)))
                for l in range(lmax + 1)
            ]
        )
        irreps_in2_pt2 = e3nn_pt2.so3.Irreps(irreps_in2)
        print(irreps_in2)
        irreps_out = o3.FullTensorProduct(irreps_in1, irreps_in2).irreps_out.simplify()
        print(irreps_out)

        print("Moving data to GPU....")
        x = irreps_in1_pt2.randn((arg_parser.batch,))
        x_numpy = x.numpy()
        np.save(f"x_{lmax}.npy", x_numpy)
        x = x.to(device="cuda")
        x_e3nn = irreps_in1.randn(arg_parser.batch, -1).to(device="cuda")

        y = irreps_in2_pt2.randn((arg_parser.batch,))
        y_numpy = y.numpy()
        np.save(f"y_{lmax}.npy", y_numpy)
        y = y.to(device="cuda")
        y_e3nn = irreps_in2.randn(arg_parser.batch, -1).to(device="cuda")
        print("Done !")

        print("Initializing TP + Linear layers on device")

        tplinear_e3nn_pt2 = e3nn_pt2.nn.TensorProductLinear(
            irreps_in1_pt2, irreps_in2_pt2, batch=arg_parser.batch
        ).to(device="cuda")
        tp_e3nn = util.jit.compile(
            o3.FullyConnectedTensorProduct(irreps_in1, irreps_in2, irreps_out).to(
                device="cuda"
            )
        )

        print("Done")

        e3x_torch_times.append(
            run(TPLinear_E3NNPT2, tplinear_e3nn_pt2, x, y, mode=arg_parser.mode)
            / arg_parser.runs
        )
        e3nn_torch_times.append(
            run(TPLinear_E3NNPT, tp_e3nn, x_e3nn, y_e3nn, mode=arg_parser.mode)
            / arg_parser.runs
        )

    if arg_parser.mode == "jax":
        x_numpy = np.load(f"x_{lmax}.npy")
        x_jax = jnp.asarray(x_numpy)

        y_numpy = np.load(f"y_{lmax}.npy")
        y_jax = jnp.asarray(y_numpy)

        rng = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(rng, 2)
        irreps_in1 = e3nn_jax.Irreps(
            [
                (arg_parser.features, (l, (-1) ** (0 if arg_parser.all_even else l)))
                for l in range(lmax + 1)
            ]
        )
        print(irreps_in1)
        irreps_in2 = e3nn_jax.Irreps(
            [
                (1, (l, (-1) ** (0 if arg_parser.all_even else l)))
                for l in range(lmax + 1)
            ]
        )
        print(irreps_in2)

        x_irreps_e3nn = e3nn_jax.normal(irreps_in1, key1, (arg_parser.batch,))
        y_irreps_e3nn = e3nn_jax.normal(irreps_in2, key2, (arg_parser.batch,))

        tplinear_e3nn = TPLineare3nn(irreps_out=irreps_in1)
        kernel = tplinear_e3nn.init(key1, x_irreps_e3nn, y_irreps_e3nn)

        e3nn_jax_times.append(
            run(
                TPLinear_E3NN,
                kernel,
                x_irreps_e3nn,
                y_irreps_e3nn,
                mode=arg_parser.mode,
            )
            / arg_parser.runs
        )

        tplinear_e3x = TPLineare3x(features=arg_parser.features, max_degree=lmax)
        kernel = tplinear_e3x.init(key1, x_jax, y_jax)

        e3x_times.append(
            run(TPLinear_E3X, kernel, x_jax, y_jax, mode=arg_parser.mode)
            / arg_parser.runs
        )


if arg_parser.plot:
    if arg_parser.mode == "torch":
        import json

        with open("e3nn_e3x_torch2.json", "w") as fp:
            json.dump(e3x_torch_times, fp)

        with open("e3nn.json", "w") as fp:
            json.dump(e3nn_torch_times, fp)

    if arg_parser.mode == "jax":
        import json

        with open("e3nn_e3x_torch2.json", "r") as fp:
            e3x_torch_times = json.load(fp)

        with open("e3nn.json", "r") as fp:
            e3nn_torch_times = json.load(fp)

        import matplotlib.pyplot as plt

        plt.plot(
            range(1, arg_parser.lmax + 1), e3x_times, "red", marker="o", label="e3x"
        )
        plt.plot(
            range(1, arg_parser.lmax + 1),
            e3nn_jax_times,
            "blue",
            marker="o",
            label="e3nn-jax",
        )
        plt.plot(
            range(1, arg_parser.lmax + 1),
            e3nn_torch_times,
            "pink",
            marker="o",
            label="e3nn",
        )
        plt.plot(
            range(1, arg_parser.lmax + 1),
            e3x_torch_times,
            "orange",
            marker="o",
            label="e3nn-pt2",
        )
        plt.legend()
        plt.xlabel("lmax")
        plt.ylabel("Time (s)")
        plt.yscale("log")
        plt.title(
            f"RTX A5500 FWD + BWD TP({arg_parser.features}x0e + {arg_parser.features}x1e ... \otimes 1x0e + 1x1e...)\n + Linear({arg_parser.features}x0e + {arg_parser.features}x1e ...)"
        )
        all_even = "_all_even" if arg_parser.all_even else ""
        plt.savefig(f"benchmark_tplinear_batch_{arg_parser.batch}{all_even}.png")
