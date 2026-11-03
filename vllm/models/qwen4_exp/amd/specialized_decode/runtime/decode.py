# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.8-Flash-Next specialized decode executor (step mode): one routing
decision per step, then the whole forward on one of two kernel families.

Built by the model state once the weights are loaded (``Qwen4ExpModelState``):
checks the loaded model, builds the per-rank launch state of each family and
compiles every kernel build of every width, before memory profiling and graph
capture. ``bind_caches`` checks the GDN states vLLM binds (the CUDA-graph
memory-profiling cache, then the real one). Each step ``plan`` returns a
``Step`` or None, and ``run`` runs a planned step's forward. Weight reloads are
not supported.

A step of at most ``MONO_MAX_TOKENS`` rows runs the mono kernels
(``mono_launch``): four launches a layer (five on GDN layers), each streaming
its weights over the whole grid with mailbox hand-offs between its stages,
which is what a few rows need:

    attention HC mix  -> front.hc_front
    linear_attn       -> blocks.gdn_block (up to out_proj's input)
    self_attn (QSA)   -> vLLM's, up to o_proj's input (``mono_launch.qsa_core``)
    out_proj / o_proj -> blocks.oproj_block (+ its all-reduce + the attention
                         HC's combine)
    MLP HC mix        -> front.hc_front
    mlp               -> blocks.moe_block (+ its all-reduce + the MLP HC's
                         combine)

A wider step of a FULL graph capture size runs the wide kernels
(``wide_launch``): batch-tiled phases separated by grid barriers, one launch
per layer segment (``wide.layer``):

    attn_mix   attention HC mix -> x
    gdn        GDN layers: projection + conv, recurrence + gated norm
    (vLLM)     QSA layers: qkv_proj and QSA attention (``wide_launch.qsa_core``)
    attn_out   out_proj / o_proj + all-reduce + the attention HC's combine
    mlp        MLP HC mix, router, MoE + all-reduce + the MLP HC's combine

The PLE layers, the final mixer and the embedding stay vLLM's. Stock defers
each HC combine to the next mix; both families materialize it with the same
bf16 rounding, so the PLE layers and the final mixer read a plain state."""

import os
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from vllm.distributed import get_tp_group
from vllm.forward_context import (
    get_forward_context,
    in_piecewise_cudagraph,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.models.qwen4_exp.amd.specialized_decode.mono import layout as L
from vllm.models.qwen4_exp.amd.specialized_decode.mono.abi import FAILED as MONO_FAILED
from vllm.models.qwen4_exp.amd.specialized_decode.runtime import NAME, DecodeConfig
from vllm.models.qwen4_exp.amd.specialized_decode.runtime import (
    mono_launch as mono,
)
from vllm.models.qwen4_exp.amd.specialized_decode.runtime import (
    wide_launch as wide,
)
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.guards import (
    FailStop,
    cache_reasons,
    model_reasons,
)
from vllm.models.qwen4_exp.amd.specialized_decode.runtime.routing import step_reason
from vllm.models.qwen4_exp.amd.specialized_decode.wide import layer as K
from vllm.models.qwen4_exp.amd.specialized_decode.wide.abi import FAILED as WIDE_FAILED

logger = init_logger(__name__)


class Step(NamedTuple):
    """A planned step: ``width`` rows, one decode token a request, on the wide
    kernels or the mono ones."""

    width: int
    attn_metadata: dict[str, Any]  # by layer name
    wide: bool


def agree(reasons: list[str], group, what: str) -> None:
    """Raises on every TP rank if any rank has ``reasons``: a rank that refused
    alone would leave the others spinning in-kernel."""
    from vllm.models.qwen4_exp.amd.specialized_decode.runtime.peer import tp_agree

    if not tp_agree(not reasons, group):
        why = "; ".join(reasons) if reasons else "another TP rank refused"
        raise ValueError(
            f"{NAME} does not support {what}: {why} (without --monokernel, "
            "vLLM's decode runs)"
        )


def compile_mono(build, key) -> None:
    """Compiles one mono launch's kernel without running it: FlyDSL's
    compile-only mode returns before the launch. ``build``: the module's
    ``build``."""
    launch = build(key)
    # FlyDSL's compile-only switch has no FLYDSL_ prefix
    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        launch(*_mono_zeros(key), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old


def _mono_zeros(key) -> list[int]:
    from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks, front

    abi = {
        front.FrontBuild: front.ABI,
        blocks.GdnBuild: blocks.GDN_ABI,
        blocks.OprojBuild: blocks.OPROJ_ABI,
        blocks.MoeBuild: blocks.MOE_ABI,
    }[type(key)]
    return [0] * len(abi.names)


class SpecializedDecode:
    """The executor over one ``Qwen4ExpModel``: builds the launch state, binds
    it to each KV cache and plans and runs every decode step."""

    def __init__(self, model, config: DecodeConfig):
        tp = get_tp_group()
        agree(model_reasons(model, tp.world_size), tp.cpu_group, "the loaded model")
        self.model, self.config = model, config
        self.gdn_prefix = next(
            ly.linear_attn.prefix
            for ly in model.layers
            if ly.layer_type == "linear_attention"
        )
        dev = model.embed_tokens.weight.device
        group = (tp.world_size, tp.rank_in_group, tp.cpu_group, dev)
        self.mono = mono.Launcher(config.mono_widths, *group)
        mono_builds = self._mono_builds()
        for build, key in mono_builds:
            compile_mono(build, key)
        failed = [self.mono.health[MONO_FAILED : MONO_FAILED + 1]]
        self.wide: wide.Launcher | None = None
        wide_builds: list = []
        if config.wide_widths:
            self.wide = wide.Launcher(model, config.wide_widths, *group)
            wide_builds = sorted(
                {k for s in config.wide_widths for k in self._wide_keys(s)}, key=repr
            )
            for key in wide_builds:
                K.compile_only(key)
            failed.append(self.wide.health[WIDE_FAILED : WIDE_FAILED + 1])
        self.failstop = FailStop(failed, tp.rank_in_group, tp.world_size, tp.cpu_group)
        self._kv: tuple[int, int] | None = None
        self.steps = 0
        # the epochs at the last LOG_EVERY step, copied without blocking
        self.counted = torch.zeros(2, dtype=torch.int32, pin_memory=True)
        self.counted_event: torch.cuda.Event | None = None
        self.counted_at = 0
        logger.info(
            "%s: %d layers; mono widths 1-%d, %d builds; wide widths %s, %d "
            "builds, %d launches a step (%s), %s TP all-reduce lines; other steps "
            "run vLLM's decode",
            NAME,
            len(model.layers),
            max(config.mono_widths),
            len(mono_builds),
            ",".join(map(str, config.wide_widths)) or "none",
            len(wide_builds),
            self.wide.launches if self.wide else 0,
            "split" if self.wide is None or self.wide.split else "fused",
            self.wide.allreduce if self.wide else "-",
        )

    def _mono_builds(self) -> list:
        from vllm.models.qwen4_exp.amd.specialized_decode.mono import blocks, front

        la = self.mono
        keys: set[tuple[Callable[[Any], Any], Any]] = set()
        for s in self.config.mono_widths:
            keys.add((blocks.build_moe, blocks.MoeBuild(tokens=s, tp=la.tp)))
            for layer in self.model.layers:
                for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection):
                    for m in (False, True):
                        keys.add((front.build, la.front_key(hc, s, m)))
                if layer.layer_type == "linear_attention":
                    keys.add((blocks.build_gdn, la.gdn_key(layer, s)))
                keys.add((blocks.build_oproj, la.oproj_key(layer, s)))
        return sorted(keys, key=lambda bk: repr(bk[1]))

    def _wide_keys(self, rows: int) -> list:
        """Every layer's wide builds at ``rows``, in step order."""
        assert self.wide is not None
        out, stats = [], True
        for layer in self.model.layers:
            stats |= layer.ple is not None
            out += self.wide.keys(layer, rows, stats)
            stats = False
        return out

    def _kv_key(self) -> tuple[int, int] | None:
        layer = next(
            ly for ly in self.model.layers if ly.layer_type == "linear_attention"
        )
        state = layer.linear_attn.kv_cache
        if not isinstance(state, (list, tuple)) or len(state) != 2:
            return None
        if state[1].numel() == 0:
            return None
        return state[1].data_ptr(), state[1].numel()

    def bind_caches(self) -> None:
        """Checks the GDN states vLLM just bound (the CUDA-graph memory
        profiling cache, then the real one) and binds them."""
        self._kv = None
        if self._kv_key() is None:
            return
        tp = get_tp_group()
        agree(cache_reasons(self.model, tp.world_size), tp.cpu_group, "the KV caches")
        if self.wide is not None:
            self.wide.bind_caches()
        self._kv = self._kv_key()

    def plan(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors,
    ) -> Step | None:
        """This step's plan, or None to run vLLM's forward.

        Steps that run vLLM's forward:
        - pipeline-parallel stage inputs, and steps before a KV cache is bound
          (profiling, dummy runs);
        - PIECEWISE steps: they run (or capture) the compiled stock model's
          graphs, and taking one would leave them uncaptured for the prefills
          of that size;
        - widths without a build (``DecodeConfig.widths``);
        - steps that are not pure decode steps (``step_reason``).

        Eager steps of a built width run the kernels too. Every predicate is the
        same on all TP ranks: the kernels' mailboxes and exchanges need every
        rank on one path. Under FULL graphs this runs at capture only and the
        plan is baked in; a replay with fewer requests than its graph pads rows
        with NULL_BLOCK_ID state slots, whose GDN state the kernels leave
        alone."""
        if intermediate_tensors is not None or self._kv is None or input_ids is None:
            return None
        if not is_forward_context_available() or in_piecewise_cudagraph():
            return None
        width = input_ids.size(0)
        if width not in self.config.widths:
            return None
        md = get_forward_context().attn_metadata
        if step_reason(md, self.gdn_prefix, width) is not None:
            return None
        assert isinstance(md, dict)
        if self._kv_key() != self._kv:
            raise RuntimeError(
                f"{NAME}: the KV cache was re-bound without bind_caches "
                "(ModelState.initialize_kv_cache)"
            )
        is_wide = width in self.config.wide_widths
        logger.info_once(
            "%s: first specialized step (%d rows, %s kernels)",
            NAME,
            width,
            "wide" if is_wide else "mono",
        )
        return Step(width, md, is_wide)

    def run(
        self,
        step: Step,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        query_start_loc: torch.Tensor | None,
        ngram_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """``Qwen4ExpModel.forward`` of a planned step -> the final mixer's
        hidden states [S, H]. ``inputs_embeds``: the multimodal wrapper's
        embeddings of the step's tokens (decode rows are sampled text tokens,
        so the token embeddings), else None."""
        model = self.model
        if inputs_embeds is None:
            inputs_embeds = model.embed_input_ids(input_ids)
        state = inputs_embeds.repeat(1, L.HC)
        args = (step, state, input_ids, positions, query_start_loc, ngram_context)
        state = self._run_wide(*args) if step.wide else self._run_mono(*args)
        _, h, _ = model.hyper_connection_mixer.mix(state)
        return h

    def _run_mono(
        self, step, state, input_ids, positions, query_start_loc, ngram_context
    ):
        la, md = self.mono, step.attn_metadata
        la.epoch.add_(1)
        msq: torch.Tensor | None = None
        for layer in self.model.layers:
            if layer.ple is not None:
                state = layer.ple(state, input_ids, query_start_loc, ngram_context)
                msq = None
            x, inj = la.hc_front(
                layer, layer.attn_hyper_connection, state, msq, la.attn_scratch
            )
            gate = None
            if layer.layer_type == "linear_attention":
                core = la.gdn(layer, md, x)
            else:
                core, gate = mono.qsa_core(layer.self_attn, x, positions)
            state, msq = la.oproj(layer, core, gate, state, inj)
            x, inj = la.hc_front(
                layer, layer.mlp_hyper_connection, state, msq, la.mlp_scratch
            )
            state, msq = la.moe(layer, x, state, inj)
        return state

    def _run_wide(
        self, step, state, input_ids, positions, query_start_loc, ngram_context
    ):
        la, md, rows = self.wide, step.attn_metadata, step.width
        assert la is not None
        la.epoch.add_(1)
        x = la.x[:rows]
        seq, stats = 0, True
        for i, layer in enumerate(self.model.layers):
            if layer.ple is not None:
                state = layer.ple(state, input_ids, query_start_loc, ngram_context)
                stats = True
            dyn = {"state": state, "x": x}
            if layer.layer_type == "linear_attention":
                gdn_md = md[layer.linear_attn.prefix]
                dyn["st_idx"] = gdn_md.non_spec_state_indices_tensor
            for key in la.keys(layer, rows, stats):
                if key.segments[0] == "attn_out" and key.attn == "qsa":
                    attn, gate = wide.qsa_core(layer.self_attn, x, positions)
                    dyn |= {"attn": attn, "gate": gate}
                seq = la.launch(key, i, seq, dyn)
            stats = False
        return state

    LOG_EVERY = 1024

    def after_step(self) -> None:
        """Once per executed step, from compute_logits (eager on every rank,
        also under graph replay). Raises on every rank once a rank gave up an
        in-kernel wait; every LOG_EVERY steps logs how many of the steps up to
        the previous LOG_EVERY ran on each kernel family (the epochs count
        specialized steps, graph replays included; copied without a host sync,
        so the counts trail by LOG_EVERY steps)."""
        self.failstop.after_step()
        if torch.cuda.is_current_stream_capturing():
            return
        self.steps += 1
        if self.steps % self.LOG_EVERY:
            return
        if self.counted_event is not None and self.counted_event.query():
            logger.info(
                "%s: %d mono and %d wide steps of %d",
                NAME,
                int(self.counted[0]),
                int(self.counted[1]),
                self.counted_at,
            )
        self.counted[0:1].copy_(self.mono.epoch[:1], non_blocking=True)
        if self.wide is not None:
            self.counted[1:2].copy_(self.wide.epoch[:1], non_blocking=True)
        self.counted_event = torch.cuda.Event()
        self.counted_event.record()
        self.counted_at = self.steps
