# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU equivalence of the Muse Spark token-sharded residual stream."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("The token-sharded path is ROCm-only", allow_module_level=True)

from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.models.muse_spark.amd import model as muse_model
from vllm.models.muse_spark.amd import muse_moe_runner
from vllm.models.muse_spark.common.layers import (
    materialize_residual_path_constants,
    norm_then_gated_residual,
)
from vllm.transformers_utils.configs.muse_spark import MuseSparkTextConfig
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from ..utils import multi_gpu_test

NUM_LAYERS = 2
HIDDEN_SIZE = 256
# TP-only sharding requires T % tp == 0; under EP, 301 also covers the padding.
NUM_TOKENS = {True: (301, 512), False: (302, 512)}


def _all_reduce_reference():
    """Keep the reference off the reduce-scatter finalize that the sharded path uses."""
    return patch.object(
        muse_moe_runner.MuseSparkTopKWeightAndReduce,
        "rs_min_tokens",
        property(lambda self: 1 << 30, lambda self, value: None),
        create=True,
    )


class _TokenMixingAttention(nn.Module):
    """Mixes all tokens like attention does, without a KV cache."""

    def __init__(self, *args, prefix: str = "", **kwargs) -> None:
        super().__init__()
        self.layer_name = prefix

    def forward(self, q, k, v):
        return q + q.float().mean(0).to(q.dtype)


def _make_config(expert_parallel: bool) -> VllmConfig:
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            enable_expert_parallel=expert_parallel,
            all2all_backend="allgather_reducescatter",
        ),
    )
    config.model_config = SimpleNamespace(
        architecture="MuseSpark1xForCausalLM",
        is_moe=True,
        sleep_mode_offload_cudagraph=False,
        max_model_len=4096,
        hf_text_config=MuseSparkTextConfig(
            hidden_act="silu",
            attention_bias=False,
            rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
            num_hidden_layers=NUM_LAYERS,
            layer_types=["sliding_attention", "full_attention"],
            mlp_layer_types=["sparse"] * NUM_LAYERS,
            layer_rope_theta=[500000.0, 0.0],
            sliding_window=128,
            max_position_embeddings=4096,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            qk_scale_factor=5.47,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            residual_gate_temperature=0.3,
            num_local_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=0,
            moe_hidden_size=128,
            moe_intermediate_size=128,
            routed_scaling_factor=1.0,
            output_multiplier=0.18,
        ),
        dtype=torch.bfloat16,
    )
    assert muse_model.token_sharded_stream_supported(config)
    return config


def _randomize(module: nn.Module, rank: int) -> None:
    """Identical weights on every rank plus distinct rank shards.

    Distinct rank contributions expose missing or duplicate reductions.
    """
    torch.manual_seed(7)
    for param in module.parameters():
        param.normal_(std=0.08)
    for name, sub in module.named_modules():
        if name.endswith("o_proj"):
            sub.weight.add_(rank * 0.01)
        if name.endswith("experts.routed_experts"):
            sub.w13_weight.add_(rank * 0.01)
            sub.quant_method.process_weights_after_loading(sub)
    materialize_residual_path_constants(module)


def _check_moe_block(rank: int, config: VllmConfig) -> None:
    with set_current_vllm_config(config):
        block = muse_model.MuseSpark1xMoE(
            config, layer_idx=0, prefix="model.layers.0.feed_forward"
        )
    _randomize(block, rank)
    for num_tokens in NUM_TOKENS[config.parallel_config.enable_expert_parallel]:
        torch.manual_seed(num_tokens)
        x = torch.randn(num_tokens, HIDDEN_SIZE)
        with (
            set_current_vllm_config(config),
            set_forward_context(None, config, num_tokens=num_tokens),
        ):
            shard = muse_model.TokenShard(num_tokens)
            actual = block(shard.local(x), shard).clone()
            with _all_reduce_reference():
                expected = shard.local(block(x))
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.02)


def _run_layers(layers, positions, x, shard) -> torch.Tensor:
    if shard is not None:
        x = shard.local(x)
    for layer in layers:
        x = layer(positions, x, shard)
    out = norm_then_gated_residual(x)
    if shard is not None:
        out = shard.gather(out)
    return out.clone()


def _check_decoder_layers(rank: int, config: VllmConfig) -> None:
    with (
        set_current_vllm_config(config),
        patch.object(muse_model, "Attention", _TokenMixingAttention),
        patch.object(muse_model, "supports_fused_kv_insert", lambda attn: False),
    ):
        layers = nn.ModuleList(
            muse_model.MuseSpark1xDecoderLayer(config, prefix=f"model.layers.{i}")
            for i in range(NUM_LAYERS)
        )
    _randomize(layers, rank)
    for num_tokens in NUM_TOKENS[config.parallel_config.enable_expert_parallel]:
        torch.manual_seed(num_tokens)
        x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.float32)
        positions = torch.arange(num_tokens)
        with (
            set_current_vllm_config(config),
            set_forward_context(None, config, num_tokens=num_tokens),
        ):
            with _all_reduce_reference():
                expected = _run_layers(layers, positions, x, None)
            actual = _run_layers(
                layers, positions, x, muse_model.TokenShard(num_tokens)
            )
        # The reduction order differs, so bf16 rounding moves single elements by
        # ~1 ulp; a dropped or duplicated reduction is off by more than 10%.
        error = (actual - expected).norm() / expected.norm()
        assert error < 1e-2, f"relative error {error:.2e}"


def _run(rank: int, port: int, check, expert_parallel: bool) -> None:
    if not expert_parallel:
        # Without EP, token sharding is enabled by the bf16 MoE reduction.
        os.environ["VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE"] = "bf16"
    torch.accelerator.set_device_index(rank)
    torch.set_num_threads(1)
    config = _make_config(expert_parallel)
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        init_workspace_manager(torch.device(f"cuda:{rank}"))
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=2)
            with torch.device(f"cuda:{rank}"), torch.inference_mode():
                torch.set_default_dtype(torch.bfloat16)
                check(rank, config)
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("expert_parallel", [True, False], ids=["tp2_ep", "tp2"])
def test_token_sharded_moe_block_matches_replicated(expert_parallel: bool) -> None:
    """The token-sharded MoE block matches its replicated-stream path."""
    mp.spawn(
        _run,
        args=(get_open_port(), _check_moe_block, expert_parallel),
        nprocs=2,
    )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("expert_parallel", [True, False], ids=["tp2_ep", "tp2"])
def test_token_sharded_decoder_layers_match_replicated(expert_parallel: bool) -> None:
    """Sharded attention, boundary and MoE match the replicated residual stream."""
    mp.spawn(
        _run,
        args=(get_open_port(), _check_decoder_layers, expert_parallel),
        nprocs=2,
    )
