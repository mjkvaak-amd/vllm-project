# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The wide decode's launches. A decoder layer is four segments, each a run of
grid phases (``SEGMENTS``):

    attn_mix   [stats] attn mix -> x
    gdn        GDN proj + conv -> recurrence + gated norm -> core (GDN layers)
    attn_out   out_proj (GDN, MXFP4) or gated o_proj (QSA, bf16)
               -> TP exchange -> attn combine
    mlp        mlp mix -> router -> route + quantize -> experts
               -> TP exchange -> mlp combine

A build (``LayerBuild.segments``) is any contiguous run of them, so how a
layer is cut into launches is the caller's choice (``layer_plan``): one launch
per segment, or as few as vLLM's QSA attention allows (it runs between
``attn_mix`` and ``attn_out``).

Phases are separated by grid barriers, a launch's last phase by the launch
boundary; an exchange's readers wait on its line flags instead. A launch reads
the multi-stream state, updates it in place (``state``, [rows, HC * HIDDEN]
bf16, materialized: the next launch, a PLE layer and the final mixer read it
as is) and leaves ``rrms`` of every (row, stream) in its scratch for the next
mix; ``stats`` computes them first when the state comes from vLLM (layer 0,
after a PLE layer). Everything a segment hands the next one is in ``x`` or the
scratch, so a cut between segments changes no value.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64

from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.grid import (
    Grid,
    TraceInvariant,
    grid_sync,
    prologue,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.ops import (
    ld,
    ld_ptr64,
    memrealtime,
    st,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.common.plan import (
    KernelAbi,
    key_tuple,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.layout import (
    BLOCKS,
    CONV_DIM,
    HC,
    HIDDEN,
    LOWRANK,
    MAX_TOKENS,
    QSA_CORE,
    SLOTS,
    THREADS,
    VALUE_DIM,
)
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import exchange as ex_
from vllm.models.qwen4_exp.amd.specialized_decode.wide.stages import gdn, hc, moe, tiled

SEGMENTS = ("attn_mix", "gdn", "attn_out", "mlp")
ATTNS = ("gdn", "qsa")  # the attention block of a layer
ATTN, MLP = 0, 1  # a layer's two exchanges


@dataclass(frozen=True)
class LayerBuild:
    # the launch runs the segments first..last of SEGMENTS (the build key holds
    # scalars only, ``plan.key_tuple``)
    first: str
    last: str
    tokens: int
    tp: int
    # the layer's attention: "gdn" or "qsa" for builds whose code depends on it
    # (``gdn``, ``attn_out``), "" otherwise, so such builds are shared
    attn: str = ""
    stats: bool = False  # the input state came from vLLM: compute its rrms first
    qkvz_shuffled: bool = False  # in_proj_qkvz on AITER's ASM FP4 GEMM layout
    out_shuffled: bool = False  # out_proj likewise
    rip: int = 384  # the experts' padded intermediate rows a rank
    hc_eps: float = 1e-6
    gdn_eps: float = 1e-6
    trace: bool = False  # CTA 0 stamps the clock at every phase end (``phase_names``)
    trace_at: int = 0  # the first stamp's word in the layer's trace row
    allreduce: str = "bf16"  # the TP exchange's line codec (``exchange.CODECS``)

    @property
    def segments(self) -> tuple[str, ...]:
        return SEGMENTS[SEGMENTS.index(self.first) : SEGMENTS.index(self.last) + 1]

    def __post_init__(self):
        ends = (self.first, self.last)
        if not set(ends) <= set(SEGMENTS) or not self.segments:
            raise ValueError(f"segments {ends}: a run of {SEGMENTS}")
        segs = self.segments
        needs_attn = "gdn" in segs or "attn_out" in segs
        if needs_attn != (self.attn in ATTNS) or (self.attn == "qsa" and "gdn" in segs):
            raise ValueError(f"segments {segs} with attention {self.attn!r}")
        if self.attn == "qsa" and "attn_mix" in segs and "attn_out" in segs:
            raise ValueError("vLLM's QSA attention runs between attn_mix and attn_out")
        if self.stats and segs[0] != "attn_mix":
            raise ValueError("stats belong to a build that starts at attn_mix")
        if self.allreduce not in ex_.CODECS:
            raise ValueError(f"all-reduce codec {self.allreduce!r}")
        if self.tokens % ex_.TOKENS:
            raise ValueError(f"{self.tokens} rows: a multiple of {ex_.TOKENS}")


def layer_plan(attn: str, split: bool) -> list[tuple[str, ...]]:
    """A layer's builds in order: one segment each (``split``), else as few as
    vLLM's attention allows (a GDN layer in one, a QSA layer in two)."""
    segs = [s for s in SEGMENTS if attn == "gdn" or s != "gdn"]
    if split:
        return [(s,) for s in segs]
    if attn == "gdn":
        return [tuple(segs)]
    return [("attn_mix",), ("attn_out", "mlp")]


ABI = KernelAbi(
    (
        "state",
        "x",
        "scratch",
        "a_norm",
        "a_di",
        "a_up",
        "m_norm",
        "m_di",
        "m_up",
        "w_qkvz",
        "w_qkvz_s",
        "w_ba",
        "w_conv",
        "a_log",
        "dt_bias",
        "gdn_norm",
        "w_out",
        "w_out_s",
        "conv_state",
        "conv_stride",
        "ssm",
        "ssm_stride",
        "st_idx",
        "attn",
        "gate",
        "w_gate",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "peers",
        "rank",
        "epoch",
        "layer",
        "sync",
        "sync_next",
        "folds",
        "health",
        "alarm",
        "trace",
    )
)
TRACE_WORDS = 32  # i64 stamps a layer in the trace buffer


def phase_names(key) -> list[str]:
    """The stamps of a traced build (``LayerBuild.trace``), in order: each
    marks the end of the named phase ("start": the kernel's start)."""
    mix = lambda t: [f"{t}.mix_down", f"{t}.mix_reduce", f"{t}.mix_up"]  # noqa: E731
    by_segment = {
        "attn_mix": (["stats"] if key.stats else []) + mix("attn"),
        "gdn": ["gdn_proj+conv", "gdn_recur"],
        "attn_out": ["out+allreduce+combine"],
        "mlp": mix("mlp")
        + ["router", "route+quant", "experts", "moe_combine+allreduce+combine"],
    }
    return ["start"] + [n for s in key.segments for n in by_segment[s]]


# --------------------------------------------------------------------------
# scratch: a width's intermediate buffers at static offsets
# --------------------------------------------------------------------------


def _part_words(rows: int) -> int:
    plans = [
        hc.down_plan(),
        gdn.proj_plan(),
        gdn.out_plan(),
        gdn.out_plan(QSA_CORE),
        moe.router_plan(),
    ]
    return max(p.partial_words(rows) for p in plans)


def scratch_layout(rows: int) -> dict[str, tuple[int, int]]:
    """Name -> (byte offset, bytes) of a width's scratch, 256-byte aligned.
    ``split`` (the MoE's split-unit marks) comes first and is sized for the
    widest step, so every width's marks share one zeroed region that nothing
    else writes."""
    sizes = {
        "split": MAX_TOKENS * SLOTS * 4,
        "rrms": rows * HC * 4,
        "inj_a": rows * HC * 4,
        "inj_m": rows * HC * 4,
        "down": rows * LOWRANK * 2,
        "qkv": rows * CONV_DIM * 2,
        "z": rows * VALUE_DIM * 2,
        "ba": gdn.ba_plan().partial_words(rows) * 4,
        "core": rows * VALUE_DIM * 2,
        "part": _part_words(rows) * 4,
        "route": 2 * rows * SLOTS * 4,
        "x4": rows * HIDDEN // 8 * 4,
        "x4s": rows * HIDDEN // 32 * 4,
        "mpart": rows * SLOTS * HIDDEN * 4,
        "mpart2": rows * SLOTS * HIDDEN * 4,
    }
    out, off = {}, 0
    for name, n in sizes.items():
        out[name] = (off, n)
        off = (off + n + 255) // 256 * 256
    return out


def scratch_bytes(rows: int) -> int:
    return max(o + n for o, n in scratch_layout(rows).values())


class Scratch(TraceInvariant):
    """A width's scratch buffers (``scratch_layout``) as byte addresses."""

    rrms: fx.Int64
    inj_a: fx.Int64
    inj_m: fx.Int64
    down: fx.Int64
    qkv: fx.Int64
    z: fx.Int64
    ba: fx.Int64
    core: fx.Int64
    part: fx.Int64
    route: fx.Int64
    x4: fx.Int64
    x4s: fx.Int64
    mpart: fx.Int64
    mpart2: fx.Int64
    split: fx.Int64

    def __init__(self, base, rows: int):
        for name, (off, _) in scratch_layout(rows).items():
            setattr(self, name, fx.Int64(base) + fx.Int64(off))


# --------------------------------------------------------------------------
# the kernels
# --------------------------------------------------------------------------

_BUILDS: dict = {}
_STREAM = fx.Stream(None)


def build(key: LayerBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    S, tp = key.tokens, key.tp
    shape = moe.Shape(S, key.rip)
    mw = moe.lds_words(shape)
    aux_words = max(hc.up_lds_words(S), ex_.GroupTile.words(S), gdn.recur_lds_words())

    @fx.struct
    class GemmLds:
        stage: fx.Array[fx.Int32, tiled.stage_words(S), 16]
        aux: fx.Array[fx.Int32, aux_words, 16]

    @fx.struct
    class MoeLds:
        index: fx.Array[fx.Int32, mw.index, 16]
        units: fx.Array[fx.Int32, mw.units, 16]
        scan: fx.Array[fx.Int32, mw.scan, 16]
        x: fx.Array[fx.Int32, mw.x, 16]
        xs: fx.Array[fx.Int32, mw.xs, 16]
        act: fx.Array[fx.Int32, mw.act, 16]
        acts: fx.Array[fx.Int32, mw.acts, 16]
        stage: fx.Array[fx.Int32, mw.stage, 16]

    @fx.union
    class PhaseLds:
        gemm: GemmLds
        moe: MoeLds

    @fx.struct
    class Smem:
        word: fx.Array[fx.Int32, 4, 16]

    keyed = key_tuple(key)
    name = f"q4w_{'_'.join(key.segments)}{('_' + key.attn) if key.attn else ''}"
    name += f"_s{S}_tp{tp}" + ("_st" if key.stats else "")
    name += "" if key.allreduce == "bf16" else f"_{key.allreduce}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        state: Int64,
        x: Int64,
        scratch: Int64,
        a_norm: Int64,
        a_di: Int64,
        a_up: Int64,
        m_norm: Int64,
        m_di: Int64,
        m_up: Int64,
        w_qkvz: Int64,
        w_qkvz_s: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        gdn_norm: Int64,
        w_out: Int64,
        w_out_s: Int64,
        conv_state: Int64,
        conv_stride: Int64,
        ssm: Int64,
        ssm_stride: Int64,
        st_idx: Int64,
        attn: Int64,
        gate: Int64,
        w_gate: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
        alarm: Int64,
        trace: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        sm = alloc.allocate(Smem).peek()
        ph = alloc.allocate(PhaseLds)
        gl, ml = ph.gemm.peek(), ph.moe.peek()
        g = Grid(tid, bid, sync, sync_next, folds, health, alarm, sm.word.ptr)
        prologue(g)
        scr = Scratch(scratch, S)
        aux_f32 = fx.recast_iter(fx.Float32, gl.aux.ptr)
        gemm = tiled.Gemm(gl.stage.ptr, S, tid)
        up_gemm = tiled.Gemm(gl.stage.ptr, S, tid, k_total=LOWRANK)
        bases = [ld_ptr64(peers, r) for r in range(tp)]
        own = bases[0]
        for r in fx.range_constexpr(1, tp):
            own = (rank == r).select(bases[r], own)
        ep = ld(epoch, 0, fx.Int32)
        names: list[str] = []

        def stamp(name):
            if const_expr(key.trace):
                if (bid == 0) & (tid == 0):
                    at = layer * TRACE_WORDS + key.trace_at + len(names)
                    st(trace, at, memrealtime())
                names.append(name)

        def end_phase(name, last=False):
            # the launch's last phase ends at the launch boundary; a traced build
            # takes the barrier so CTA 0's stamp marks every CTA's end
            if const_expr(not last or key.trace):
                grid_sync(g)
            stamp(name)

        stamp("start")

        def exchange(kind: int):
            slot = (layer % 2) * 2 + kind
            tag = ep * 128 + layer * 2 + kind + 1
            return ex_.Exchange(bases, own, rank, tp, slot, tag, S, key.allreduce)

        def mix(tag, norm, di, up, inj, last):
            hc.mix_down(g, gemm, state, scr.rrms, norm, di, scr.part)
            end_phase(f"{tag}.mix_down")
            hc.mix_reduce(g, scr.part, scr.down, inj, S, True)
            end_phase(f"{tag}.mix_reduce")
            hc.mix_up(g, up_gemm, aux_f32, scr.down, up, state, scr.rrms, norm, x, S)
            end_phase(f"{tag}.mix_up", last)

        def back(weights, source, k):
            xch = exchange(ATTN)
            tile = ex_.GroupTile(gl.aux.ptr, S)
            gdn.out(g, gemm, weights, source, scr.part, tile, xch, k, 0)
            hc.combine(g, xch, state, scr.inj_a, scr.rrms, S, key.hc_eps)
            end_phase("out+allreduce+combine", key.segments[-1] == "attn_out")

        def mlp_half():
            mix("mlp", m_norm, m_di, m_up, scr.inj_m, False)
            moe.router(g, gemm, w_gate, x, scr.part)
            end_phase("router")
            moe.quantize_rows(g, x, scr.x4, scr.x4s, S)
            moe.route(g, scr.part, scr.route, S)
            end_phase("route+quant")
            lds = moe.Lds(
                index=ml.index.ptr,
                units=ml.units.ptr,
                scan=ml.scan.ptr,
                x=ml.x.ptr,
                xs=ml.xs.ptr,
                act=ml.act.ptr,
                acts=ml.acts.ptr,
                stage=ml.stage.ptr,
            )
            xch = exchange(MLP)
            moe.experts(
                g,
                shape,
                lds,
                scr.route,
                scr.x4,
                scr.x4s,
                w13,
                w13s,
                w2,
                w2s,
                scr.mpart,
                scr.mpart2,
                scr.split,
                xch.tag,
            )
            end_phase("experts")
            moe.combine(g, xch, scr.route, scr.mpart, scr.mpart2, scr.split, S)
            hc.combine(g, xch, state, scr.inj_m, scr.rrms, S, key.hc_eps)
            end_phase("moe_combine+allreduce+combine", True)

        if const_expr("attn_mix" in key.segments):
            if const_expr(key.stats):
                hc.stats(g, state, scr.rrms, S, key.hc_eps)
                end_phase("stats")
            mix("attn", a_norm, a_di, a_up, scr.inj_a, key.segments[-1] == "attn_mix")
        if const_expr("gdn" in key.segments):
            w = gdn.Bufs(
                qkvz=w_qkvz,
                qkvz_s=w_qkvz_s,
                shuffled=key.qkvz_shuffled,
                ba=w_ba,
                conv=w_conv,
                a_log=a_log,
                dt_bias=dt_bias,
                norm=gdn_norm,
            )
            b = gdn.Bufs(
                x=x,
                qkv=scr.qkv,
                z=scr.z,
                ba=scr.ba,
                core=scr.core,
                part=scr.part,
                st_idx=st_idx,
                conv_state=conv_state,
                conv_stride=conv_stride,
                ssm=ssm,
                ssm_stride=ssm_stride,
            )
            gdn.proj(g, gemm, w, b, S, 0)
            end_phase("gdn_proj+conv")
            gdn.recur(g, w, b, aux_f32, S, key.gdn_eps)
            end_phase("gdn_recur", key.segments[-1] == "gdn")
        if const_expr("attn_out" in key.segments):
            if const_expr(key.attn == "gdn"):
                out_w = tiled.Fp4Weights(
                    w_out, w_out_s, HIDDEN, VALUE_DIM, key.out_shuffled
                )
                back(out_w, tiled.Source(scr.core, VALUE_DIM), VALUE_DIM)
            else:
                back(
                    tiled.Bf16Weights(w_out, HIDDEN, QSA_CORE),
                    gdn.GatedCore(attn, gate, QSA_CORE),
                    QSA_CORE,
                )
        if const_expr("mlp" in key.segments):
            mlp_half()
        if const_expr(key.trace):
            assert names == phase_names(key), (names, phase_names(key))

    @flyc.jit
    def launch(
        state: Int64,
        x: Int64,
        scratch: Int64,
        a_norm: Int64,
        a_di: Int64,
        a_up: Int64,
        m_norm: Int64,
        m_di: Int64,
        m_up: Int64,
        w_qkvz: Int64,
        w_qkvz_s: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        gdn_norm: Int64,
        w_out: Int64,
        w_out_s: Int64,
        conv_state: Int64,
        conv_stride: Int64,
        ssm: Int64,
        ssm_stride: Int64,
        st_idx: Int64,
        attn: Int64,
        gate: Int64,
        w_gate: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        layer: Int32,
        sync: Int64,
        sync_next: Int64,
        folds: Int64,
        health: Int64,
        alarm: Int64,
        trace: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kernel(
            state,
            x,
            scratch,
            a_norm,
            a_di,
            a_up,
            m_norm,
            m_di,
            m_up,
            w_qkvz,
            w_qkvz_s,
            w_ba,
            w_conv,
            a_log,
            dt_bias,
            gdn_norm,
            w_out,
            w_out_s,
            conv_state,
            conv_stride,
            ssm,
            ssm_stride,
            st_idx,
            attn,
            gate,
            w_gate,
            w13,
            w13s,
            w2,
            w2s,
            peers,
            rank,
            epoch,
            layer,
            sync,
            sync_next,
            folds,
            health,
            alarm,
            trace,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kernel, launch)
    _BUILDS[key] = launch
    return launch


def run(key: LayerBuild, args: dict) -> None:
    """Launch ``key``'s kernel on the current stream; ``args``: every ABI name
    (tensors by data pointer, ints as is)."""
    from vllm.v1.attention.ops.flydsl_kernels import _run_compiled

    vals = [
        v.data_ptr() if isinstance(v, torch.Tensor) else int(v) for v in ABI.pack(args)
    ]
    _run_compiled(build(key), *vals, torch.cuda.current_stream())


def compile_only(key: LayerBuild) -> None:
    """Compile ``key``'s kernel without launching it (FlyDSL's compile-only
    mode returns before the launch)."""
    import os

    old = os.environ.get("COMPILE_ONLY")
    os.environ["COMPILE_ONLY"] = "1"
    try:
        f = build(key)
        f(*ABI.zeros(), stream=torch.cuda.current_stream())
    finally:
        if old is None:
            del os.environ["COMPILE_ONLY"]
        else:
            os.environ["COMPILE_ONLY"] = old
