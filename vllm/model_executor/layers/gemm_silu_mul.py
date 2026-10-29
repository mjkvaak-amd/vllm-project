# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merged gate_up GEMM + SiluAndMul in one Triton kernel.

``silu_and_mul(x @ W.T)`` with ``W = [W_gate; W_up]`` of shape [2N, K]. Each
program loads an A tile once and multiplies it with the matching row blocks of
W_gate and W_up (two views of the merged weight), so the [M, 2N] intermediate is
never written and the activation runs on the accumulators.

Whether the fused kernel is used is decided per (M, N, K, dtype) by timing it
against the unfused path (the platform GEMM followed by ``silu_and_mul``) on
first use outside of graph capture; the faster one is kept.
"""

import torch

import vllm._custom_ops  # noqa: F401 - registers torch.ops._C.silu_and_mul
import vllm.model_executor.layers.utils  # noqa: F401 - registers rocm_unquantized_gemm
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def _gemm_silu_mul_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_wn,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (rm % M).to(tl.int64)[:, None] * stride_xm + rk[None, :]
    g_ptrs = w_ptr + (rn % N).to(tl.int64)[None, :] * stride_wn + rk[:, None]
    u_ptrs = g_ptrs + N.to(tl.int64) * stride_wn
    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(x_ptrs)
            g = tl.load(g_ptrs)
            u = tl.load(u_ptrs)
        else:
            k_mask = rk < K - k * BLOCK_K
            a = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
            g = tl.load(g_ptrs, mask=k_mask[:, None], other=0.0)
            u = tl.load(u_ptrs, mask=k_mask[:, None], other=0.0)
        acc_g = tl.dot(a, g, acc_g)
        acc_u = tl.dot(a, u, acc_u)
        x_ptrs += BLOCK_K
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K
    # Round like the unfused GEMM output so both paths compute the same thing.
    dtype = out_ptr.dtype.element_ty
    g = acc_g.to(dtype).to(tl.float32)
    u = acc_u.to(dtype).to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * u
    o_ptrs = out_ptr + rm.to(tl.int64)[:, None] * stride_om + rn[None, :]
    tl.store(o_ptrs, y.to(dtype), mask=(rm < M)[:, None] & (rn < N)[None, :])


_LDS_BYTES = 64 * 1024


def _candidates(M: int, N: int, K: int, itemsize: int) -> list[dict]:
    """Weight-streaming configs: one M tile, narrow N tiles, long K steps."""
    BM = max(16, triton.next_power_of_2(M))
    out = []
    for BN in (16, 32, 64, 128):
        if BN > max(16, triton.next_power_of_2(N)):
            continue
        for BK in (64, 128, 256, 512):
            if BK > max(64, triton.next_power_of_2(K)) or BN * BK > 64 * 512:
                continue
            if BM * BN > 64 * 64 and BK > 128:
                continue
            for warps in (4, 8) if BN * BK >= 32 * 512 else (4,):
                for stages in (2, 3):
                    if BK * (BM + 2 * BN) * itemsize * stages > _LDS_BYTES:
                        continue
                    out.append(
                        dict(
                            BLOCK_M=BM,
                            BLOCK_N=BN,
                            BLOCK_K=BK,
                            num_warps=warps,
                            num_stages=stages,
                        )
                    )
    return out


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == x.dtype
        and weight.dim() == 2
        and weight.shape[0] % 2 == 0
        and weight.stride(1) == 1
        and x.shape[-1] == weight.shape[1]
        and x.stride(-1) == 1
    )


def gemm_silu_mul_triton(
    x: torch.Tensor, weight: torch.Tensor, cfg: dict, out: torch.Tensor | None = None
) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = weight.shape[0] // 2
    if out is None:
        out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    cfg = dict(cfg)
    kw = {"matrix_instr_nonkdim": 16} if current_platform.is_rocm() else {}
    grid = (triton.cdiv(N, cfg["BLOCK_N"]), triton.cdiv(M, cfg["BLOCK_M"]))
    _gemm_silu_mul_kernel[grid](
        x2,
        weight,
        out,
        M,
        N,
        K,
        x2.stride(0),
        weight.stride(0),
        out.stride(0),
        EVEN_K=K % cfg["BLOCK_K"] == 0,
        **cfg,
        **kw,
    )
    return out.reshape(*x.shape[:-1], N)


def gemm_silu_mul_unfused(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if current_platform.is_rocm():
        y = torch.ops.vllm.rocm_unquantized_gemm(x, weight, None)
    else:
        y = torch.nn.functional.linear(x, weight)
    out = torch.empty((*y.shape[:-1], y.shape[-1] // 2), dtype=y.dtype, device=y.device)
    torch.ops._C.silu_and_mul(out, y)
    return out


# (M, N, K, dtype, stride_wn) -> Triton config, or None for the unfused path
_choice: dict[tuple, dict | None] = {}
_bench_stream: torch.cuda.Stream | None = None


def _bench(fn, variants: int = 1, reps: int = 20, iters: int = 7) -> float:
    """Microseconds per call: the median over ``iters`` replays of a graph of
    ``reps`` back-to-back calls ``fn(i % variants)``, so that host launch
    overhead does not decide the comparison."""
    global _bench_stream
    reps = max(reps, variants)
    if _bench_stream is None:
        _bench_stream = torch.cuda.Stream()
    stream = _bench_stream
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            fn(i % variants)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i in range(reps):
                fn(i % variants)
        graph.replay()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(iters + 1)]
        events[0].record(stream)
        for i in range(iters):
            graph.replay()
            events[i + 1].record(stream)
    events[-1].synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    times = sorted(events[i].elapsed_time(events[i + 1]) for i in range(iters))
    return times[iters // 2] * 1000.0 / reps


# In a model every layer streams its own weight, so it is cold in the
# last-level cache (256 MiB on MI300/MI350); time with enough rotating weight
# copies to exceed it.
_COLD_BYTES = 512 * 1024 * 1024


def _weight_copies(weight: torch.Tensor) -> list[torch.Tensor]:
    nbytes = weight.numel() * weight.element_size()
    n = min(16, -(-_COLD_BYTES // nbytes))
    free, _ = torch.cuda.mem_get_info(weight.device)
    n = max(1, min(n, int(free * 0.5) // nbytes))
    return [weight] + [weight.clone() for _ in range(n - 1)]


def _tune(x: torch.Tensor, weight: torch.Tensor) -> dict | None:
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = weight.shape[0] // 2
    ws = _weight_copies(weight)
    ref = gemm_silu_mul_unfused(x2, weight)
    t_ref = _bench(lambda i: gemm_silu_mul_unfused(x2, ws[i]), len(ws))
    out = torch.empty_like(ref)
    best, t_best = None, t_ref
    for cfg in _candidates(M, N, K, x.element_size()):
        try:
            gemm_silu_mul_triton(x2, weight, cfg, out)
            if not torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2):
                continue
            t = _bench(
                lambda i, c=cfg: gemm_silu_mul_triton(x2, ws[i], c, out), len(ws)
            )
        except Exception as e:  # e.g. out of shared memory on this device
            logger.debug("gemm_silu_mul config %s failed: %s", cfg, e)
            continue
        if t < t_best:
            best, t_best = cfg, t
    logger.info(
        "gemm_silu_mul M=%d N=%d K=%d: unfused %.1f us, %s",
        M,
        N,
        K,
        t_ref,
        f"fused {t_best:.1f} us {best}" if best else "keeping unfused",
    )
    return best


def gemm_silu_mul_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if not supported(x, weight):
        return gemm_silu_mul_unfused(x, weight)
    M = x.numel() // x.shape[-1]
    key = (M, weight.shape[0] // 2, weight.shape[1], x.dtype, weight.stride(0))
    if key not in _choice:
        if torch.cuda.is_current_stream_capturing():
            return gemm_silu_mul_unfused(x, weight)
        _choice[key] = _tune(x, weight)
    cfg = _choice[key]
    if cfg is None:
        return gemm_silu_mul_unfused(x, weight)
    return gemm_silu_mul_triton(x, weight, cfg)


def gemm_silu_mul_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0] // 2))


direct_register_custom_op(
    op_name="gemm_silu_mul",
    op_func=gemm_silu_mul_impl,
    fake_impl=gemm_silu_mul_fake,
)
