# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next specialized decode on MI355X (TP2, ``--monokernel``),
step mode: FlyDSL persistent kernels for each decoder layer on pure decode
steps of 1 to 128 rows.

``hooks`` holds the model's entry points (``SpecializedDecodeMixin``; the
model runner state that builds the executor is ``qwen4_exp/amd/model_state.py``),
``runtime`` the refusals, the plan of each step and the executor, and the
launches of the two kernel families: ``mono`` for steps of at most 8 rows (one
launch per layer block, mailbox hand-offs between the stages of a block) and
``wide`` for the wider FULL graph sizes (batch-tiled phases behind grid
barriers, one launch per layer segment). Tests live in
``tests/models/qwen4_exp/amd_specialized_decode/``. See ``README.md``."""
