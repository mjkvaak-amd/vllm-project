# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-sized TP all-reduce over AITER's mesh-lending transport.

On an xGMI full mesh a TP group of 2 or 4 GPUs uses 1 or 3 of each GPU's 7
links. AITER's `MeshLendAllReduce` also moves the payload through the GPUs
outside the group, on their SDMA engines. vLLM decides here which calls take
it and with which payload.
"""

import torch
from torch.distributed import ProcessGroup

from vllm._aiter_ops import rocm_aiter_ops
from vllm.logger import init_logger

logger = init_logger(__name__)

MB = 1024 * 1024


def mesh_lend_policy(world_size: int, quantization: str) -> tuple[int, int] | None:
    """(payload bits, smallest payload in bytes) for a TP size and a
    QuickReduce regime, or None where the mesh path does not pay.

    The payload never loses more than the regime the operator chose: exact
    unless QuickReduce runs INT4, and then 4 bits with each operand
    quantized once. From each size up the mesh path was measured faster
    than the backend that takes the call otherwise (MI355X and MI325X, the
    GPUs outside the group idle).
    """
    if world_size == 2 and quantization == "NONE":
        return 0, 4 * MB
    if world_size == 2 and quantization == "INT4":
        return 4, 24 * MB
    if world_size == 4 and quantization == "NONE":
        return 0, 40 * MB
    # TP4 with INT4: QuickReduce is faster at every size. FP, INT8, INT6
    # and INT3 were not measured against.
    return None


class MeshLendAllReduce:
    # Largest payload. AITER holds 2.5 times this per rank at TP2, 2 times at TP4.
    MAX_SIZE: int = 256 * MB

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        world_size: int,
        quantization: str,
    ) -> None:
        self.disabled = True
        policy = mesh_lend_policy(world_size, quantization)
        if policy is None:
            logger.info(
                "Mesh-lending all-reduce is off for TP%d with QuickReduce %s.",
                world_size,
                quantization,
            )
            return
        self.bits, self.min_size = policy
        try:
            self._impl = rocm_aiter_ops.create_mesh_lend_all_reduce(
                group=group, device=device, max_bytes=self.MAX_SIZE
            )
        except RuntimeError as e:
            # AITER raises on every rank when any rank cannot set it up.
            logger.warning("Mesh-lending all-reduce is disabled: %s", e)
            return
        self.disabled = False
        logger.info(
            "Mesh-lending all-reduce: %s payload from %d MiB, %d lender GPUs.",
            f"INT{self.bits}" if self.bits else "exact",
            self.min_size // MB,
            self._impl.num_lenders,
        )

    def should_use(self, inp: torch.Tensor) -> bool:
        """Same answer on every rank: dtype, layout and size only. A graph
        cannot hold the call, which posts packets from the host."""
        if self.disabled or inp.dtype != torch.bfloat16:
            return False
        if not self.min_size <= inp.numel() * 2 <= self.MAX_SIZE:
            return False
        return inp.is_contiguous() and not torch.cuda.is_current_stream_capturing()

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._impl.all_reduce(inp, bits=self.bits)

    def close(self) -> None:
        if not self.disabled:
            self.disabled = True
            self._impl.close()
