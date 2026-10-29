# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused gate_up GEMM + SiluAndMul vs the unfused path, per decode batch size.

    python benchmarks/kernels/benchmark_gemm_silu_mul.py [--shapes 2048x8192 ...]

A shape is HIDDENxINTERMEDIATE (K x N); the merged weight is [2N, K].
"""

import argparse

import torch

from vllm.model_executor.layers import gemm_silu_mul as g

DEFAULT_SHAPES = [
    "2048x8192",  # Llama-3.2-1B
    "2048x6144",  # Qwen3-1.7B
    "3072x8192",  # Llama-3.2-3B
    "4096x14336",  # Llama-3.1-8B
    "4096x12288",  # Qwen3-8B
    "5120x17408",  # Qwen3-14B / 32B (TP1 slice of 25600 at TP>1 differs)
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=DEFAULT_SHAPES)
    p.add_argument("--m", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)
    print(f"{'K x N':>12} {'M':>4} {'unfused us':>11} {'fused us':>9} {'speedup':>8}  config")
    for shape in args.shapes:
        K, N = map(int, shape.split("x"))
        w = torch.randn(2 * N, K, device="cuda", dtype=dtype) / K**0.5
        for M in args.m:
            x = torch.randn(M, K, device="cuda", dtype=dtype)
            ref = g.gemm_silu_mul_unfused(x, w)
            ws = g._weight_copies(w)
            t_ref = g._bench(lambda i: g.gemm_silu_mul_unfused(x, ws[i]), len(ws))
            cfg = g._tune(x, w)
            if cfg is None:
                print(f"{shape:>12} {M:>4} {t_ref:>11.1f} {'-':>9} {'1.00':>8}  unfused")
                continue
            out = g.gemm_silu_mul_triton(x, w, cfg)
            err = (out.float() - ref.float()).abs().max().item()
            t = g._bench(lambda i: g.gemm_silu_mul_triton(x, ws[i], cfg, out), len(ws))
            print(
                f"{shape:>12} {M:>4} {t_ref:>11.1f} {t:>9.1f} {t_ref / t:>8.2f}  "
                f"{cfg} maxerr={err:.3g}"
            )


if __name__ == "__main__":
    main()
