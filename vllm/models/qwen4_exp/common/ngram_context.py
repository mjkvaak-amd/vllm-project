# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step Qwen4Exp PLE inputs built in a single launch.

The n-gram context of a request is the ``ctx_len`` tokens that precede its
first scheduled token, with EOS where the request has fewer than ``ctx_len``
tokens before it. Rows ``>= num_reqs`` (CUDA-graph padding) are all EOS.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ngram_context_kernel(
    context_ptr,
    context_stride,
    qsl_out_ptr,
    qsl_in_ptr,
    idx_mapping_ptr,
    num_computed_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    num_reqs,
    num_rows,
    num_qsl_in,
    num_qsl_out,
    eos_token_id,
    CTX_LEN: tl.constexpr,
    BLOCK_CTX: tl.constexpr,
    BLOCK_QSL: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offs = tl.arange(0, BLOCK_CTX)
    mask = offs < CTX_LEN
    tokens = tl.full((BLOCK_CTX,), eos_token_id, tl.int32)
    if pid < num_reqs:
        req = tl.load(idx_mapping_ptr + pid).to(tl.int64)
        end = tl.load(num_computed_ptr + req).to(tl.int64)
        pos = end - CTX_LEN + offs
        valid = mask & (pos >= 0)
        loaded = tl.load(
            all_token_ids_ptr + req * all_token_ids_stride + pos,
            mask=valid,
            other=eos_token_id,
        )
        tokens = loaded.to(tl.int32)
    tl.store(
        context_ptr + pid * context_stride + offs, tokens, mask=mask & (pid < num_rows)
    )

    # query_start_loc[i] = src[min(i, num_qsl_in - 1)], spread over the grid.
    last = num_qsl_in - 1
    for start in range(pid * BLOCK_QSL, num_qsl_out, num_programs * BLOCK_QSL):
        i = start + tl.arange(0, BLOCK_QSL)
        src = tl.load(qsl_in_ptr + tl.minimum(i, last), mask=i < num_qsl_out)
        tl.store(qsl_out_ptr + i, src, mask=i < num_qsl_out)


def prepare_ngram_context(
    context: torch.Tensor,
    query_start_loc_out: torch.Tensor,
    query_start_loc_in: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
    num_reqs: int,
    eos_token_id: int,
) -> None:
    """Fill ``context`` and ``query_start_loc_out`` in one launch.

    ``context`` is ``[num_reqs_padded, ctx_len]`` int32 and is fully written.
    ``query_start_loc_out`` gets ``query_start_loc_in``, with its last entry
    repeated if ``query_start_loc_out`` is longer.
    """
    num_rows, ctx_len = context.shape
    assert context.dtype == torch.int32 and context.stride(1) == 1
    assert all_token_ids.stride(1) == 1
    assert 0 <= num_reqs <= num_rows
    assert query_start_loc_in.numel() >= 1
    # One program even without rows, to write query_start_loc.
    _ngram_context_kernel[(max(num_rows, 1),)](
        context,
        context.stride(0),
        query_start_loc_out,
        query_start_loc_in,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        num_reqs,
        num_rows,
        query_start_loc_in.numel(),
        query_start_loc_out.numel(),
        eos_token_id,
        CTX_LEN=ctx_len,
        BLOCK_CTX=triton.next_power_of_2(ctx_len),
        BLOCK_QSL=128,
        num_warps=1,
    )
