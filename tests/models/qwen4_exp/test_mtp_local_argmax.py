# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``get_top_tokens`` on the Qwen4Exp MTP drafter.

The speculator opts into ``use_local_argmax_reduction`` by probing the draft
model for ``get_top_tokens``, and expects it to select the same greedy draft
token as the full-vocab argmax over ``compute_logits``.
"""

from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("Tests the ROCm Qwen4Exp MTP drafter.", allow_module_level=True)

from vllm.models.qwen4_exp.amd.mtp import Qwen4ExpMTP  # noqa: E402

DEVICE_TYPE = current_platform.device_type
HIDDEN_SIZE = 256
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"


def _make_drafter(vocab_size: int) -> Qwen4ExpMTP:
    # A single vocab shard, so the reduction returns the local argmax without
    # touching the process group.
    with (
        mock.patch(f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=0),
        mock.patch(
            f"{_EMBEDDING}.get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        lm_head = ParallelLMHead(vocab_size, HIDDEN_SIZE, prefix="lm_head")
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        lm_head.weight.normal_(0.0, 0.02, generator=generator)

    # Only the attributes the two methods read.
    drafter = object.__new__(Qwen4ExpMTP)
    torch.nn.Module.__init__(drafter)
    drafter.lm_head = lm_head.to(device=DEVICE_TYPE, dtype=torch.bfloat16)
    drafter.logits_processor = LogitsProcessor(vocab_size)
    return drafter


def _hidden_states(num_tokens: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, generator=generator)
    return hidden_states.to(device=DEVICE_TYPE, dtype=torch.bfloat16)


def _assert_selects_max_logit(top: torch.Tensor, logits: torch.Tensor) -> None:
    """The selected token must carry the row's maximum logit (bf16 rows can
    tie, so compare the selected values rather than the indices)."""
    assert top.dtype == torch.int64
    assert top.shape == (logits.shape[0],)
    selected = logits.gather(1, top.unsqueeze(1)).squeeze(1)
    assert torch.equal(selected, logits.max(dim=-1).values)


@pytest.mark.parametrize("vocab_size", [512, 500])
def test_get_top_tokens_matches_full_argmax(default_vllm_config, vocab_size):
    drafter = _make_drafter(vocab_size)
    for num_tokens in (1, 16, 128):
        hidden_states = _hidden_states(num_tokens, seed=11)
        logits = drafter.compute_logits(hidden_states)
        top = drafter.get_top_tokens(hidden_states)
        _assert_selects_max_logit(top, logits)
        assert (top < vocab_size).all()
