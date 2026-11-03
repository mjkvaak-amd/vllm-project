# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next specialized decode on MI355X (TP2), enabled by
``--monokernel`` (``KernelConfig.monokernel``).

``specialized_decode_config`` runs in the model's ``__init__``, before the
weights load. It returns None for the stock decode, and raises with every
reason from ``unsupported`` when ``--monokernel`` is set for a deployment the
kernels cannot serve (default-deny). Checks that need the loaded model run when
the executor is built (``decode.SpecializedDecode``), those that need the GDN
states at every KV cache bind (``decode.SpecializedDecode.bind_caches``)."""

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layout as W

logger = init_logger(__name__)

TP = 2
NAME = "Qwen3.8-Flash-Next specialized decode (--monokernel)"
# the widest step the mono kernels serve; wider steps go to the wide kernels.
# At TP2 on MI355X the mono kernels are ahead up to 8 rows and the wide ones
# from 12 (median ITL, 1K and 8K inputs)
MONO_MAX_TOKENS = 8


@dataclass(frozen=True)
class DecodeConfig:
    """``mono_widths``: every row count up to MONO_MAX_TOKENS (eager steps and
    FULL graph capture sizes alike; a graph's pad rows are inert).
    ``wide_widths``: the deployment's CUDA graph capture sizes above that, up to
    the wide kernels' MAX_TOKENS, in multiples of 8. A step of another width
    runs vLLM's decode."""

    wide_widths: tuple[int, ...] = ()
    mono_widths: tuple[int, ...] = tuple(range(1, MONO_MAX_TOKENS + 1))

    @property
    def widths(self) -> tuple[int, ...]:
        return self.mono_widths + self.wide_widths


def decode_widths(vllm_config) -> tuple[int, ...]:
    """The capture sizes the wide kernels serve."""
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or ()
    return tuple(
        sorted(
            s
            for s in set(sizes)
            if MONO_MAX_TOKENS < s <= W.MAX_TOKENS and s % W.MIN_TOKENS == 0
        )
    )


def specialized_decode_config(vllm_config) -> DecodeConfig | None:
    """The deployment's specialized decode settings, None for the stock decode.
    Raises when ``--monokernel`` is set for an unsupported deployment, listing
    every reason."""
    if not vllm_config.kernel_config.monokernel:
        return None
    reasons = unsupported(vllm_config)
    if reasons:
        raise ValueError(
            f"{NAME} does not support: " + "; ".join(reasons) + " (without "
            "--monokernel, vLLM's decode runs)"
        )
    if piecewise_graphs_only(vllm_config):
        logger.warning(
            "%s: PIECEWISE-only CUDA graphs capture every decode step as stock; "
            "use FULL decode graphs",
            NAME,
        )
    return DecodeConfig(wide_widths=decode_widths(vllm_config))


def unsupported(vllm_config) -> list[str]:
    """Every reason this deployment cannot run the specialized decode (empty: it
    can). Config only: the loaded model is checked when the executor is built.
    Default-deny: every feature below is refused, the kernels serve none of
    them."""
    from vllm import envs
    from vllm.config import CompilationMode
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
    )
    from vllm.models.qwen4_exp.common.hyperconnection import (
        is_hc_sequence_parallel_enabled,
    )

    mc, pc, cc = (
        vllm_config.model_config,
        vllm_config.parallel_config,
        vllm_config.cache_config,
    )
    oc = vllm_config.offload_config
    checks = (
        (pc.tensor_parallel_size != TP, f"TP {pc.tensor_parallel_size} (TP{TP} only)"),
        (pc.pipeline_parallel_size != 1, "pipeline parallelism"),
        (pc.data_parallel_size != 1, "data parallelism"),
        (pc.enable_expert_parallel, "expert parallelism"),
        (pc.enable_eplb, "EPLB"),
        (pc.decode_context_parallel_size != 1, "decode context parallelism"),
        (pc.prefill_context_parallel_size != 1, "prefill context parallelism"),
        # the kernels read one batch's attention metadata per step
        (pc.use_ubatching, "DBO (dual-batch overlap)"),
        # the kernels take and return every token row on every rank
        (
            is_hc_sequence_parallel_enabled(vllm_config),
            "sequence parallelism (HC or MoE)",
        ),
        (vllm_config.lora_config is not None, "LoRA"),
        (vllm_config.kv_transfer_config is not None, "KV transfer connectors"),
        (
            vllm_config.aux_output_config.enable_return_routed_experts,
            "routed-expert capture",
        ),
        # the kernels read the loaded weights in place
        (mc.enable_sleep_mode, "sleep mode"),
        (
            oc.uva.cpu_offload_gb > 0 or oc.prefetch.offload_group_size > 0,
            "CPU weight offload",
        ),
        (vllm_config.weight_transfer_config is not None, "weight updates"),
        # the GDN kernels run plain decode rows, one token a request
        (vllm_config.speculative_config is not None, "speculative decoding"),
        # the GDN kernels read and write one state slot a request in place
        (
            cc.mamba_cache_mode != "none",
            (
                f"GDN prefix caching (mamba_cache_mode {cc.mamba_cache_mode!r}; "
                "pass --no-enable-prefix-caching)"
            ),
        ),
        (
            MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                mc.dtype, cc.mamba_cache_dtype, cc.mamba_ssm_cache_dtype
            )
            != (L.CONV_DTYPE, L.STATE_DTYPE),
            (
                "GDN cache dtypes other than a BF16 conv state and an FP32 "
                "recurrent state (mamba cache dtype)"
            ),
        ),
        (
            cc.cache_dtype not in ("auto", "bfloat16"),
            f"QSA KV cache dtype {cc.cache_dtype!r} (BF16 only)",
        ),
        # the MoE kernels run the shared expert as routed expert 512
        (
            not envs.VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS,
            (
                "an unfused shared expert (set "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1)"
            ),
        ),
        # the model state builds the executor and binds it to each KV cache
        (
            not vllm_config.use_v2_model_runner,
            "the V1 model runner (set VLLM_USE_V2_MODEL_RUNNER=1)",
        ),
        # plan() and the launches run in forward; a compiled forward would
        # trace them
        (
            vllm_config.compilation_config.mode == CompilationMode.STOCK_TORCH_COMPILE,
            "torch.compile (compilation mode STOCK_TORCH_COMPILE)",
        ),
    )
    reasons = [why for bad, why in checks if bad]
    text = mc.hf_text_config
    for names, want in ((L.SHAPE_NAMES, L.SHAPES), (W.SHAPE_NAMES, W.SHAPES)):
        shapes = text_shapes(text, names)
        if shapes != want:
            reasons.append(
                f"model shapes {shapes} ({', '.join(names)}; "
                f"Qwen3.8-Flash-Next is {want})"
            )
    if not getattr(text, "norm_topk_prob", True):
        reasons.append("router weights without top-k renormalization")
    gate = getattr(text, "output_gate_type", "sigmoid")
    if gate != "sigmoid":
        reasons.append(f"GDN output gate {gate!r} (sigmoid only)")
    layers = getattr(text, "num_hidden_layers", 0)
    if layers > min(L.MAX_LAYERS, W.MAX_LAYERS):
        reasons.append(
            f"{layers} decoder layers (at most {min(L.MAX_LAYERS, W.MAX_LAYERS)})"
        )
    return reasons + platform_reasons()


def text_shapes(text, names=L.SHAPE_NAMES) -> tuple:
    return tuple(getattr(text, name, None) for name in names)


def platform_reasons() -> list[str]:
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return ["this platform (ROCm only)"]
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return ["this GPU (gfx950 / MI355X only)"]
    import torch

    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus != L.BLOCKS:
        # every CTA of a launch must be resident: a partitioned GPU deadlocks
        return [f"{cus} CUs (a partitioned GPU; the kernels need {L.BLOCKS})"]
    return []


def piecewise_graphs_only(vllm_config) -> bool:
    """Decode steps captured as PIECEWISE graphs only: plan() runs them stock."""
    from vllm.config import CUDAGraphMode

    mode = vllm_config.compilation_config.cudagraph_mode
    if mode is None or mode == CUDAGraphMode.NONE:
        return False
    return mode.decode_mode() != CUDAGraphMode.FULL
