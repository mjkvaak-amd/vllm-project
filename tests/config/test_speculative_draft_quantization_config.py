# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``speculative_config.quantization_config`` reaches the draft ``ModelConfig``.

The target's ``--quantization-config`` is resolved into
``QuantizationConfigArgs`` by the engine args, and the weight loader expects
that type. The draft's online quantization has to arrive resolved the same
way, and stay off when it is not set.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from transformers import PreTrainedConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.quantization import QuantizationConfigArgs
from vllm.config.speculative import SpeculativeConfig


def _draft_model_config_kwargs(**speculative_kwargs) -> dict[str, Any]:
    hf_config = SpeculativeConfig.hf_config_override(
        PreTrainedConfig(
            architectures=["DeepseekV3ForCausalLM"],
            model_type="deepseek_v3",
            num_nextn_predict_layers=1,
        )
    )
    draft = MagicMock(
        hf_config=hf_config, architectures=hf_config.architectures, max_model_len=128
    )
    draft.registry.inspect_model_cls.return_value = (None, "DeepSeekMTPModel")
    target = MagicMock(max_model_len=128, quantization=None, hf_overrides={})
    target.hf_config.model_type = "deepseek_v3"

    with patch("vllm.config.speculative.ModelConfig", return_value=draft) as model:
        SpeculativeConfig(
            model="draft",
            method="mtp",
            num_speculative_tokens=1,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
            **speculative_kwargs,
        )
    return dict(model.call_args.kwargs)


@pytest.mark.cpu_test
def test_draft_quantization_config_is_resolved():
    targets = {"mtp.*.experts": "mxfp4"}
    kwargs = _draft_model_config_kwargs(quantization_config={"targets": targets})

    resolved = kwargs["quantization_config"]
    assert isinstance(resolved, QuantizationConfigArgs)
    assert resolved.targets == targets


@pytest.mark.cpu_test
def test_draft_quantization_config_defaults_to_none():
    assert _draft_model_config_kwargs()["quantization_config"] is None
