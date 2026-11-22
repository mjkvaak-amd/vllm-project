# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.device_communicators.mesh_lend_all_reduce import (
    MB,
    mesh_lend_policy,
)


@pytest.mark.parametrize(
    ("world_size", "quantization", "expected"),
    [
        (2, "NONE", (0, 4 * MB)),
        (2, "FP", None),
        (2, "INT4", (4, 24 * MB)),
        (2, "INT8", None),
        (2, "INT3", None),
        (4, "NONE", (0, 40 * MB)),
        (4, "INT4", None),
        (8, "NONE", None),
    ],
)
def test_mesh_lend_policy(world_size, quantization, expected):
    """Fails if the mesh path would quantize a payload the operator asked to
    reduce exactly, or take a TP size or regime it was not measured faster in."""
    assert mesh_lend_policy(world_size, quantization) == expected
