# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AR + Qwen4Exp HyperConnection combine-norm fusion.

Lives apart from ``test_fusion_all_reduce.py`` because importing the AMD
Qwen4Exp HC ops registers custom ops that collide with the NVIDIA ones in a
shared process.
"""

import pytest
import torch

from tests.compile.backend import TestBackend
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
    RocmAiterAllReduceFusionPass,
)
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type

HC_COUNT = 4
EPS = 1e-6


class TestAiterAllReduceHCCombineNormModel(torch.nn.Module):
    """``all_reduce -> qwen4_exp_hc_combine_norm``, the shape the block
    boundary produces once the row-parallel projection has reduced."""

    def __init__(
        self,
        hidden_size: int,
        token_num: int,
        hc_count: int = HC_COUNT,
        eps: float = EPS,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.eps = eps
        # Per-branch norm: one hidden_size slice of weight per stream.
        self.weight = torch.nn.Parameter(
            torch.randn(hidden_size * hc_count, dtype=dtype) / hidden_size,
            requires_grad=False,
        )

    def forward(
        self,
        block_output: torch.Tensor,
        residual: torch.Tensor,
        injection_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.models.qwen4_exp.amd.ops.hc import hc_combine_norm

        reduced = tensor_model_parallel_all_reduce(block_output)
        return hc_combine_norm(
            residual,
            reduced,
            injection_logits,
            self.weight,
            self.eps,
            self.hc_count,
        )

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.rocm_aiter_fused_allreduce_hc_combine_norm.default]


@pytest.mark.parametrize("token_num", [8, 64])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm AITER AR+HC-combine-norm fusion is ROCm-only",
)
@pytest.mark.skipif(not IS_AITER_FOUND, reason="aiter is not found")
def test_rocm_aiter_hc_combine_norm_fusion_pass_replace(
    token_num: int,
    hidden_size: int,
    dtype: torch.dtype,
    world_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

    master_port = get_open_port()
    torch.multiprocessing.spawn(
        _hc_combine_norm_fusion_pass_on_test_model,
        args=(world_size, master_port, token_num, hidden_size, dtype),
        nprocs=world_size,
    )


def _hc_combine_norm_fusion_pass_on_test_model(
    local_rank: int,
    world_size: int,
    master_port: int,
    token_num: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    set_random_seed(0)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_CUSTOM_AR": "1",
        }
    )
    rocm_aiter_ops.refresh_env_variables()

    init_distributed_environment()

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE)
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    vllm_config.parallel_config.rank = local_rank
    vllm_config.model_config = ModelConfig(
        model="RedHatAI/Llama-3.2-1B-Instruct-FP8",
        trust_remote_code=True,
        dtype=dtype,
        seed=42,
    )
    # The pass keys the HC pattern off the model config; the stand-in model
    # above is not a real Qwen4Exp checkpoint.
    vllm_config.model_config.hf_config.hc_count = HC_COUNT
    vllm_config.model_config.hf_config.hc_per_branch_norm = True

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        all_reduce_fusion_pass = RocmAiterAllReduceFusionPass(vllm_config)
        if all_reduce_fusion_pass.disabled:
            pytest.skip("RocmAiterAllReduceFusionPass is disabled in this build")

        backend = TestBackend(
            NoOpEliminationPass(vllm_config),
            all_reduce_fusion_pass,
            FixFunctionalizationPass(vllm_config),
            PostCleanupPass(vllm_config),
        )

        model = TestAiterAllReduceHCCombineNormModel(
            hidden_size, token_num, dtype=dtype
        )
        block_output = torch.randn((token_num, hidden_size), requires_grad=False)
        residual = torch.randn((token_num, hidden_size * HC_COUNT), requires_grad=False)
        # Keep the gate off sigmoid's saturated tails so the comparison is
        # sensitive to the epilogue rather than to a clamped scale.
        injection_logits = torch.randn((token_num, HC_COUNT), requires_grad=False)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(block_output, residual, injection_logits)

        assert all_reduce_fusion_pass.matched_count == 1, (
            f"{all_reduce_fusion_pass.matched_count=}"
        )
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=True)
        (fused_op,) = model.ops_in_model_after()
        assert backend.op_count(fused_op) == 1

        unfused = model(block_output, residual, injection_logits)
        fused = compiled_model(block_output, residual, injection_logits)

        # The fused kernel reproduces the unfused rounding boundary -- it
        # normalizes the bf16-rounded combined sum -- so the only expected
        # divergence is the custom all-reduce's accumulation order.
        for name, ref, act in zip(("combined_state", "normed"), unfused, fused):
            torch.testing.assert_close(
                ref, act, atol=1e-2, rtol=1e-2, msg=lambda s, n=name: f"{n}: {s}"
            )

        del all_reduce_fusion_pass
