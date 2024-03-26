import argparse

parser = argparse.ArgumentParser(description="Script with mode flag")
parser.add_argument('--mode', type=str, choices=['torch', 'jax'], default='jax', help="Select mode: 'torch' or 'jax'")
parser.add_argument('--batch', type=int, default=1, help="Select batch size")
parser.add_argument('--features', type=int, default=128, help="Number of feature channels")
parser.add_argument('--lmax', type=int, default=7, help="lmax for the operation")
parser.add_argument('--runs', type=int, default=1000, help="Number of timing runs for benchmarking")
parser.add_argument('--warmup', type=int, default=100, help="Number of warmup runs for benchmarking")
parser.add_argument('--all_even', type=bool, default=False, help="Choose whether irreps are all even or not")
parser.add_argument('--plot', type=bool, default=True, help="Choose whether to make plots or not")
    
arg_parser = parser.parse_args()

if arg_parser.mode == "torch":
    import torch
    from e3nn import io, o3, util
    import e3nn_pt2
    
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

    def TP_E3NNPT2(x, y):
        result = tp2(x, y)
        loss = result.sum()
        loss.backward()
        return result, loss
    
    def TP_E3NN(x, y):
        result = tp(x, y)
        loss = result.sum()
        loss.backward()
        return result, loss

if arg_parser.mode == "jax":
    import os

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false ")
    from flax import linen as nn
    import jax
    import jax.numpy as jnp
    import e3x
    import e3nn_jax
    
    import functools
    
    class TPe3x(nn.Module):
        
        max_degree: int
        features: int

        @nn.compact
        def __call__(self, x_irreps, y_irreps):
            return e3x.nn.Tensor(max_degree=self.max_degree, include_pseudotensors=True)(x_irreps, y_irreps)
        
    @jax.jit
    def TP_E3X(kernel, x, y):
        def loss_function(kernel, x, y):
            return tp_e3x.apply(kernel, x, y).sum()
    
        loss, grad = jax.value_and_grad(loss_function)(kernel, x, y)    
        return loss

    
    class TPe3nn(nn.Module):
        
        irreps_out: e3nn_jax.Irreps

        @nn.compact
        def __call__(self, x_irreps_e3nn, y_irreps_e3nn):
            return e3nn_jax.tensor_product(x_irreps_e3nn, y_irreps_e3nn)
    

    @jax.jit
    def TP_E3NNJAX(kernel, x_irreps_e3nn, y_irreps_e3nn):
        def loss_function(kernel, x, y):
            return sum(
                jnp.sum(out)
                for out in jax.tree_util.tree_leaves(tp_e3nn.apply(kernel, x, y))
            )
                    
        loss, grad = jax.value_and_grad(loss_function)(kernel, x_irreps_e3nn, y_irreps_e3nn)    
        return loss

from time import time
import numpy as np


def run(func, sample_func, *args, mode="torch"):
    
    input_args = sample_func(*args)

    print("Warmup....")        
    for _ in range(arg_parser.warmup):
        result = func(*input_args)

    print("Benchmarking")
    start = time()
    for _ in range(arg_parser.runs):
        result = func(*input_args)
        if mode == "jax":
            # Check for IrrepsArray
            if hasattr(result, 'array'):
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result.array)
            else:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
        else:
            torch.cuda.synchronize()
    end = time()

    return (end - start)

e3nn_jax_times = []
e3x_times = []
e3x_torch_times = []
e3nn_torch_times = []
roofline_times = []

#RTX A5500
PEAK_FLOPS = 34.10/2
PEAK_BW = 0.768

for lmax in range(1, arg_parser.lmax+1):

    if arg_parser.mode == "torch":
        
        def sample_e3nn_pt2(irreps_in1_pt2, irreps_in2_pt2):
            x = irreps_in1_pt2.randn((arg_parser.batch,)).to(device='cuda')
            y = irreps_in2_pt2.randn((arg_parser.batch,)).to(device='cuda')
            return x, y
        
        def sample_e3nn(irreps_in1, irreps_in2):
            x = irreps_in1.randn(arg_parser.batch, -1).to(device='cuda')
            y = irreps_in2.randn(arg_parser.batch, -1).to(device='cuda')
            return x, y
                    
        irreps_in1 = o3.Irreps([(arg_parser.features, (l, (-1)**(0 if arg_parser.all_even else l))) for l in range(lmax + 1)])
        irreps_in1_pt2 = e3nn_pt2.so3.Irreps(irreps_in1)
        print(irreps_in1)
        irreps_in2 = o3.Irreps([(1, (l, (-1)**(0 if arg_parser.all_even else l))) for l in range(lmax + 1)])
        irreps_in2_pt2 = e3nn_pt2.so3.Irreps(irreps_in2)
        print(irreps_in2)
        
        
        print("Moving data to GPU....")
        x, y = sample_e3nn_pt2(irreps_in1_pt2, irreps_in2_pt2)
        x_e3nn, y_e3nn = sample_e3nn(irreps_in1, irreps_in2)
        print("Done !")

        print("Initializing TP layers on device")
        
        global tp2 
        tp2 = e3nn_pt2.nn.TensorProduct(irreps_in1_pt2, irreps_in2_pt2, batch=arg_parser.batch).to(device='cuda')
        tp2 = torch.compile(tp2, mode="reduce-overhead", fullgraph=True)
        tp = o3.FullTensorProduct(irreps_in1, irreps_in2).to(device='cuda')
        tp = util.jit.compile(tp)

        print("Done")
        
        e3x_torch_times.append(run(TP_E3NNPT2, sample_e3nn_pt2, irreps_in1_pt2, irreps_in2_pt2, mode=arg_parser.mode)/arg_parser.runs)
        e3nn_torch_times.append(run(TP_E3NN, sample_e3nn, irreps_in1, irreps_in2, mode=arg_parser.mode)/arg_parser.runs)

    if arg_parser.mode == "jax":
            
        rng = jax.random.PRNGKey(0)
        tp_e3x = TPe3x(features=arg_parser.features, max_degree=lmax)
        
        def sample_e3x(lmax):
            rng = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(rng, 2)
            x = jax.random.normal(key1, (arg_parser.batch, 1 if arg_parser.all_even else 2, (lmax+1)**2, arg_parser.features))
            y = jax.random.normal(key2, (arg_parser.batch, 1 if arg_parser.all_even else 2, (lmax+1)**2, arg_parser.features))
            params = tp_e3x.init(rng, x, y)
            return params, x, y

        params, x_jax, y_jax = sample_e3x(lmax)

        nflops = nn.summary._get_flops(TP_E3X, params, x_jax, y_jax)
        _,weights_bytes = nn.summary._size_and_bytes(params)
        input_bytes = x_jax.nbytes + y_jax.nbytes
        output_bytes = 4*(arg_parser.batch*arg_parser.features*2*(lmax+1)**2)
        nbytes = input_bytes + output_bytes + weights_bytes
        print(f"e3x Peak FLOPs itr_time: {((nflops*1e-12) / PEAK_FLOPS)}")
        print(f"e3x Peak Bytes itr_time: {((nbytes*1e-12) / PEAK_BW)}")
        e3x_roofline_time = max(
                        ((nflops*1e-12)
                        / PEAK_FLOPS),
                        ((nbytes*1e-12)
                        / PEAK_BW)
                    )
        print("e3x Peak GFLOPS/s: ", nflops*1e-9/e3x_roofline_time)
        roofline_times.append(e3x_roofline_time)
        e3x_time = run(TP_E3X, sample_e3x, lmax, mode=arg_parser.mode)/arg_parser.runs
        print("e3x GFLOPS/s: ", nflops*1e-9/e3x_time)
        e3x_times.append(e3x_time)


        irreps_in1 = e3nn_jax.Irreps([(arg_parser.features, (l, (-1)**(0 if arg_parser.all_even else l))) for l in range(lmax + 1)])
        print(irreps_in1)
        irreps_in2 = e3nn_jax.Irreps([(1, (l, (-1)**(0 if arg_parser.all_even else l))) for l in range(lmax + 1)])
        print(irreps_in2)

        tp_e3nn = TPe3nn(irreps_out=irreps_in1)

        def sample_e3nn_jax(irreps_in1, irreps_in2):
            rng = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(rng, 2)
            x = e3nn_jax.normal(irreps_in1, key1, (arg_parser.batch,))
            y = e3nn_jax.normal(irreps_in2, key2, (arg_parser.batch,))
            params = tp_e3nn.init(rng, x, y)
            return params, x, y
        
        params, x_jax, y_jax = sample_e3nn_jax(irreps_in1, irreps_in2)
        nflops = nn.summary._get_flops(TP_E3NNJAX, params, x_jax, y_jax)
        _,weights_bytes = nn.summary._size_and_bytes(params)
        input_bytes = x_jax.array.nbytes + y_jax.array.nbytes
        output_bytes = 4*(arg_parser.batch*arg_parser.features*2*(lmax+1)**2)
        nbytes = input_bytes + output_bytes + weights_bytes
        print(f"e3nn-jax Peak FLOPs itr_time: {((nflops*1e-12) / PEAK_FLOPS)}")
        print(f"e3nn-jax Peak Bytes itr_time: {((nbytes*1e-12) / PEAK_BW)}")
        e3nn_jax_roofline_time = max(
                        ((nflops*1e-12)
                        / PEAK_FLOPS),
                        ((nbytes*1e-12)
                        / PEAK_BW)
                    )
        print("e3nn-jax Peak GFLOPS/s: ", nflops*1e-9/e3nn_jax_roofline_time)
        e3nn_jax_time = run(TP_E3NNJAX, sample_e3nn_jax, irreps_in1, irreps_in2, mode=arg_parser.mode)/arg_parser.runs
        print("e3nn-jax GFLOPS/s: ", nflops*1e-9/e3nn_jax_time)
        e3nn_jax_times.append(e3nn_jax_time)
        
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
        plt.plot(range(1, arg_parser.lmax+1), e3x_times, 'red', marker="o", label="e3x")
        plt.plot(range(1, arg_parser.lmax+1), roofline_times, 'purple', marker="o", label="theoretical roofline")
        plt.plot(range(1, arg_parser.lmax+1), e3nn_jax_times, 'blue', marker="o",label="e3nn-jax")
        plt.plot(range(1, arg_parser.lmax+1), e3nn_torch_times, "pink", marker="o",label="e3nn")
        plt.plot(range(1, arg_parser.lmax+1), e3x_torch_times, 'orange', marker="o",label="e3nn-pt2")
        plt.legend()
        plt.xlabel("lmax")
        plt.ylabel("Time (s)")
        plt.yscale("log")
        # plt.ylim(2**-15, 2**-3)
        parity = "e" if arg_parser.all_even else "o"
        plt.title(f"RTX A5500 FWD + BWD TP({arg_parser.features}x0e + {arg_parser.features}x1{parity} ... \otimes 1x0e + 1x1{parity}...)\n -> {arg_parser.features}x0e + {arg_parser.features}x1{parity} ...)")
        all_even = "_all_even" if arg_parser.all_even else ""
        plt.savefig(f"benchmark_tplinear_batch_{arg_parser.batch}{all_even}.png")
