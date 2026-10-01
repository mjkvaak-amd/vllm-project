# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.models.qwen4_exp.common.ngram_context import prepare_ngram_context
from vllm.platforms import current_platform

DEVICE = "cuda" if current_platform.is_cuda_alike() else "cpu"


def _reference(
    num_rows, ctx_len, idx_mapping, num_computed, all_token_ids, num_reqs, eos
):
    context = torch.full((num_rows, ctx_len), eos, dtype=torch.int32)
    for i in range(num_reqs):
        req = int(idx_mapping[i])
        end = int(num_computed[req])
        for j in range(ctx_len):
            pos = end - ctx_len + j
            if pos >= 0:
                context[i, j] = all_token_ids[req, pos]
    return context


@pytest.mark.parametrize("ctx_len", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "num_reqs,num_rows", [(0, 0), (0, 4), (1, 1), (3, 4), (7, 8), (64, 64)]
)
@pytest.mark.parametrize("qsl_tail", [0, 5])
def test_prepare_ngram_context_matches_reference(ctx_len, num_reqs, num_rows, qsl_tail):
    torch.manual_seed(num_reqs * 31 + num_rows * 7 + ctx_len + qsl_tail)
    max_num_reqs, max_model_len, eos = 96, 40, 151645
    all_token_ids = torch.randint(
        0, 150000, (max_num_reqs, max_model_len), dtype=torch.int32
    )
    num_computed = torch.randint(
        0, max_model_len + 1, (max_num_reqs,), dtype=torch.int32
    )
    # Short histories exercise the EOS fill at the start of a request.
    num_computed[: max_num_reqs // 4] %= ctx_len + 1
    idx_mapping = torch.randperm(max_num_reqs)[:num_reqs].to(torch.int32)
    query_lens = torch.randint(1, 9, (num_rows,), dtype=torch.int32)
    query_lens[num_reqs:] = 0
    qsl_in = torch.zeros(num_rows + 1, dtype=torch.int32)
    torch.cumsum(query_lens, 0, out=qsl_in[1:])

    context = torch.full((num_rows, ctx_len), -1, dtype=torch.int32, device=DEVICE)
    qsl_out = torch.full(
        (num_rows + 1 + qsl_tail,), -1, dtype=torch.int32, device=DEVICE
    )
    prepare_ngram_context(
        context,
        qsl_out,
        qsl_in.to(DEVICE),
        idx_mapping.to(DEVICE),
        num_computed.to(DEVICE),
        all_token_ids.to(DEVICE),
        num_reqs,
        eos,
    )

    expected = _reference(
        num_rows, ctx_len, idx_mapping, num_computed, all_token_ids, num_reqs, eos
    )
    torch.testing.assert_close(context.cpu(), expected)
    expected_qsl = torch.cat([qsl_in, qsl_in[-1:].expand(qsl_tail)])
    torch.testing.assert_close(qsl_out.cpu(), expected_qsl)


def test_prepare_ngram_context_strided_rows():
    # The model state hands in a row slice of a [max_num_reqs, ctx_len] buffer.
    eos = 7
    buf = torch.full((8, 3), -1, dtype=torch.int32, device=DEVICE)
    all_token_ids = torch.arange(40, dtype=torch.int32, device=DEVICE).view(2, 20)
    qsl_in = torch.tensor([0, 1, 2, 2], dtype=torch.int32, device=DEVICE)
    qsl_out = torch.empty(4, dtype=torch.int32, device=DEVICE)
    prepare_ngram_context(
        buf[:3],
        qsl_out,
        qsl_in,
        torch.tensor([1, 0], dtype=torch.int32, device=DEVICE),
        torch.tensor([2, 5], dtype=torch.int32, device=DEVICE),
        all_token_ids,
        2,
        eos,
    )
    expected = torch.full((8, 3), -1, dtype=torch.int32)
    expected[:3] = torch.tensor([[22, 23, 24], [7, 0, 1], [7, 7, 7]])
    torch.testing.assert_close(buf.cpu(), expected)
    torch.testing.assert_close(qsl_out.cpu(), qsl_in.cpu())


def test_amd_model_state_prepares_ngram_context():
    from vllm.models.qwen4_exp.amd.model_state import Qwen4ExpModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    model_state = object.__new__(Qwen4ExpModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.full((8, 3), -1, dtype=torch.int32, device=DEVICE)
    model_state.ple_query_start_loc = torch.full(
        (9,), -1, dtype=torch.int32, device=DEVICE
    )
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0], dtype=torch.int32, device=DEVICE),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32, device=DEVICE),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor([3, 1], dtype=torch.int32, device=DEVICE)
        ),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor(
                [[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32, device=DEVICE
            )
        ),
    )
    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    torch.testing.assert_close(
        model_inputs["query_start_loc"].cpu(),
        torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model_inputs["ngram_context"].cpu(),
        torch.tensor([[99, 99, 20], [1, 2, 3], [99, 99, 99]], dtype=torch.int32),
    )
    assert (
        model_inputs["ngram_context"].data_ptr() == model_state.ngram_context.data_ptr()
    )
