# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator

import torch
from torch import fx
from torch.fx.experimental.symbolic_shapes import statically_known_true

import vllm.model_executor.layers.gemm_silu_mul  # noqa: F401 - registers the op
from vllm.config import VllmConfig
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..fx_utils import is_func
from ..inductor_pass import InductorPass
from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)

FUSED_OP = torch.ops.vllm.gemm_silu_mul.default
SILU_MUL_OP = torch.ops._C.silu_and_mul.default
INT64_MAX = 2**63 - 1


class GemmSiluMulFusionPass(VllmInductorPass):
    """Replace ``silu_and_mul(gemm(x, W))`` with ``gemm_silu_mul(x, W)``.

    Merged column-parallel linears (gate_up_proj) run one GEMM into an [M, 2N]
    buffer that SiluAndMul then reads back. The fused op computes the gate and
    up halves from the two row blocks of W in one kernel and applies the
    activation on the accumulators. Matches both the custom-op form of
    SiluAndMul and its native form (two slices, silu, mul).
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.max_token_num = self.pass_config.gemm_silu_mul_max_token_num

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return compile_range.end <= self.max_token_num

    @staticmethod
    def _gemm_operands(node: fx.Node) -> tuple[fx.Node, fx.Node] | None:
        """(x, W) if ``node`` computes ``x @ W.T`` without bias."""
        if current_platform.is_rocm() and is_func(
            node, torch.ops.vllm.rocm_unquantized_gemm.default
        ):
            x, w = node.args[:2]
            bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
            return (x, w) if bias is None else None
        if is_func(node, torch.ops.aten.mm.default):
            x, wt = node.args
            if is_func(wt, torch.ops.aten.t.default) or (
                is_func(wt, torch.ops.aten.permute.default)
                and list(wt.args[1]) == [1, 0]
            ):
                return x, wt.args[0]
        return None

    @staticmethod
    def _last_dim(node: fx.Node, dim: int) -> bool:
        return dim in (-1, node.meta["val"].dim() - 1)

    def _native_silu_mul(self, gemm: fx.Node, d: int) -> list[fx.Node] | None:
        """``mul, silu, slice_lo, slice_hi`` of ``silu(y[..., :d]) * y[..., d:]``
        if that is all ``y`` feeds."""
        users = list(gemm.users)
        if len(users) != 2 or not all(
            is_func(u, torch.ops.aten.slice.Tensor) for u in users
        ):
            return None
        lo = hi = None
        for s in users:
            dim, start, end = (list(s.args[1:]) + [0, INT64_MAX])[:3]
            if not self._last_dim(gemm, dim):
                return None
            if start == 0 and end == d:
                lo = s
            elif start == d and end >= 2 * d:
                hi = s
        if lo is None or hi is None or len(hi.users) != 1:
            return None
        mul = next(iter(hi.users))
        if not is_func(mul, torch.ops.aten.mul.Tensor) or hi not in mul.args:
            return None
        act = mul.args[0] if mul.args[1] is hi else mul.args[1]
        chain = self._silu_chain(act, lo)
        if chain is None:
            return None
        # every intermediate is consumed only inside the matched expression
        inside = {mul, *chain}
        if any(not set(nd.users) <= inside for nd in chain if nd is not act):
            return None
        if set(act.users) != {mul}:
            return None
        return [mul, *chain, hi]

    @staticmethod
    def _silu_chain(act: fx.Node, lo: fx.Node) -> list[fx.Node] | None:
        """Nodes from ``act`` back to ``lo`` (inclusive, consumers first) if
        ``act == silu(lo)``: ``aten.silu``, its decomposition
        ``x / (1 + exp(-x))``, or ``x * sigmoid(x)``, optionally computed in
        fp32 between two dtype conversions."""
        convert = torch.ops.prims.convert_element_type.default
        chain = []
        if is_func(act, convert):
            chain.append(act)
            act = act.args[0]

        def is_x(nd: object) -> bool:
            return nd is lo or (is_func(nd, convert) and nd.args[0] is lo)

        def x_of(nd: fx.Node) -> list[fx.Node]:
            return [lo] if nd is lo else [nd, lo]

        if is_func(act, torch.ops.aten.silu.default) and is_x(act.args[0]):
            return chain + [act] + x_of(act.args[0])
        if is_func(act, torch.ops.aten.div.Tensor) and is_x(act.args[0]):
            x, den = act.args
            if (
                is_func(den, torch.ops.aten.add.Tensor)
                and den.args[1] == 1
                and is_func(den.args[0], torch.ops.aten.exp.default)
                and is_func(den.args[0].args[0], torch.ops.aten.neg.default)
                and den.args[0].args[0].args[0] is x
            ):
                e = den.args[0]
                return chain + [act, den, e, e.args[0]] + x_of(x)
        if is_func(act, torch.ops.aten.mul.Tensor):
            for x, sig in (act.args, act.args[::-1]):
                if (
                    is_x(x)
                    and is_func(sig, torch.ops.aten.sigmoid.default)
                    and sig.args[0] is x
                ):
                    return chain + [act, sig] + x_of(x)
        return None

    @staticmethod
    def _custom_silu_mul(gemm: fx.Node) -> tuple[fx.Node, fx.Node] | None:
        """(auto_functionalized, getitem) of the SiluAndMul custom op on ``gemm``."""
        users = list(gemm.users)
        if len(users) != 1:
            return None
        at = users[0]
        if not (
            is_func(at, torch.ops.higher_order.auto_functionalized)
            and at.args[0] == SILU_MUL_OP
            and at.kwargs.get("input") is gemm
        ):
            return None
        outs = [u for u in at.users if is_func(u, operator.getitem)]
        if len(outs) != len(at.users) or any(u.args[1] != 1 for u in outs):
            return None
        return at, outs[0] if outs else None

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        count = 0
        for gemm in list(graph.nodes):
            operands = self._gemm_operands(gemm)
            if operands is None:
                continue
            x, w = operands
            w_val = w.meta.get("val")
            n2 = gemm.meta["val"].shape[-1]
            if (
                w_val is None
                or w_val.dim() != 2
                or w_val.dtype not in (torch.float16, torch.bfloat16)
                or not isinstance(n2, int)
                or n2 % 2
            ):
                continue
            d = n2 // 2

            replaced: list[fx.Node] = []
            if (native := self._native_silu_mul(gemm, d)) is not None:
                target, replaced = native[0], native
            elif (custom := self._custom_silu_mul(gemm)) is not None:
                at, target = custom
                if target is None:
                    continue
                replaced = [target, at]
            else:
                continue
            if not statically_known_true(target.meta["val"].shape[-1] == d):
                continue

            with graph.inserting_before(target):
                fused = graph.call_function(FUSED_OP, (x, w))
            fused.meta["val"] = target.meta["val"]
            target.replace_all_uses_with(fused)
            for nd in replaced:
                if len(nd.users) == 0:
                    graph.erase_node(nd)
            if len(gemm.users) == 0:
                graph.erase_node(gemm)
            count += 1
        logger.debug("Fused %d GEMM + SiluAndMul pairs", count)

    def uuid(self) -> str:
        return InductorPass.hash_dict(
            {"source": InductorPass.hash_source(self), "max": self.max_token_num}
        )
