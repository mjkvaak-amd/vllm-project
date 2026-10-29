# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``compute_logits_local`` on Qwen4Exp, the opt-in for batch-sharded sampling.

The multimodal wrapper inherits ``compute_logits_local`` from Qwen3.5 and
forwards it to its language model, so the language model has to implement it.
On ROCm it must also run the specialized decode's per-step hook, which
``compute_logits`` runs and batch-sharded sampling calls instead.
"""

from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("Tests the ROCm Qwen4Exp model.", allow_module_level=True)

from vllm.models.qwen4_exp.amd.model import Qwen4ExpForCausalLM  # noqa: E402

DEVICE_TYPE = current_platform.device_type
HIDDEN_SIZE = 256
VOCAB_SIZE = 500
_EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"


def _make_model() -> Qwen4ExpForCausalLM:
    # A single vocab shard: the gathered and the local logits are the same
    # columns, so the two methods can be compared directly.
    with (
        mock.patch(f"{_EMBEDDING}.get_tensor_model_parallel_rank", return_value=0),
        mock.patch(
            f"{_EMBEDDING}.get_tensor_model_parallel_world_size", return_value=1
        ),
    ):
        lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE, prefix="lm_head")
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        lm_head.weight.normal_(0.0, 0.02, generator=generator)

    # Only the attributes the two methods read.
    model = object.__new__(Qwen4ExpForCausalLM)
    torch.nn.Module.__init__(model)
    model.lm_head = lm_head.to(device=DEVICE_TYPE, dtype=torch.bfloat16)
    model.logits_processor = LogitsProcessor(VOCAB_SIZE)
    model.specialized_decode = mock.Mock()
    return model


def test_compute_logits_local_matches_compute_logits(default_vllm_config):
    model = _make_model()
    generator = torch.Generator().manual_seed(11)
    hidden_states = torch.randn(16, HIDDEN_SIZE, generator=generator).to(
        device=DEVICE_TYPE, dtype=torch.bfloat16
    )

    logits = model.compute_logits(hidden_states)
    local = model.compute_logits_local(hidden_states)

    assert torch.equal(local[:, :VOCAB_SIZE], logits[:, :VOCAB_SIZE])
    assert model.specialized_decode.after_step.call_count == 2
