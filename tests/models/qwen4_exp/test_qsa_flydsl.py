# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.models.qwen4_exp.amd.ops.qsa_flydsl as qsa_flydsl

TOKEN_TOPK = 2048
COMPRESS_RATIO = 4
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
WIDTH = TOKEN_TOPK + COMPRESS_RATIO - 1


def _selection_inputs(rows: int, pages: int = 8, page_size: int = 16):
    return dict(
        q=torch.zeros(rows, 4, 128, dtype=torch.bfloat16),
        k_cache=torch.zeros(pages, page_size, 1, 128, dtype=torch.bfloat16),
        page_table=torch.zeros(2, pages, dtype=torch.int32),
        token_to_req=torch.zeros(rows, dtype=torch.int32),
        query_positions=torch.arange(rows, dtype=torch.int64),
        sequence_lengths=torch.full((2,), rows, dtype=torch.int32),
        token_topk=TOKEN_TOPK,
        compress_ratio=COMPRESS_RATIO,
        out=torch.empty(rows, WIDTH, dtype=torch.int32),
    )


def _attention_inputs(rows: int):
    kv = torch.zeros(4, 16, 1, 512, dtype=torch.bfloat16)
    return dict(
        q=torch.zeros(rows, 12, 256, dtype=torch.bfloat16),
        k_cache=kv[..., :256],
        v_cache=kv[..., 256:],
        logical_indices=torch.zeros(rows, WIDTH, dtype=torch.int32),
        block_table=torch.zeros(1, 4, dtype=torch.int32),
        token_to_req=torch.zeros(rows, dtype=torch.int32),
        out=torch.empty(rows, 12, 256, dtype=torch.bfloat16),
    )


class FakeQSA:
    def __init__(self, k1_reason=None, k2_reason=None):
        self.k1_reason = k1_reason
        self.k2_reason = k2_reason
        self.k1_calls = []
        self.k2_calls = []

    def qsa_indexer_selection_supported(self, token_topk, compress_ratio):
        if (token_topk, compress_ratio) != (TOKEN_TOPK, COMPRESS_RATIO):
            return "FlyDSL K1 selects 512 blocks at compress ratio 4"
        return None

    def qsa_indexer_supported(self, q, k_cache, page_table):
        return self.k1_reason

    def qsa_indexer_block_ids(
        self,
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        context_lens,
        out,
        heads,
    ):
        assert query_positions.dtype == torch.int32
        assert out.shape == (q.shape[0], BLOCK_TOPK)
        self.k1_calls.append((q.shape[0], heads))
        out.fill_(len(self.k1_calls))
        return out

    def flydsl_qsa_paged_gqa_supported(self, q, k_cache, v_cache, indices, page_table):
        return self.k2_reason

    def flydsl_qsa_paged_gqa(
        self, q, k_cache, v_cache, indices, page_table, token_to_req, out
    ):
        self.k2_calls.append(q.shape[0])
        out.fill_(1)
        return out


@pytest.fixture
def fake_qsa(monkeypatch):
    def install(**kwargs):
        fake = FakeQSA(**kwargs)
        monkeypatch.setattr(qsa_flydsl, "flydsl_qsa", lambda: fake)
        expanded = []

        def expand(blocks, positions, lengths, token_to_req, ratio, topk, out):
            expanded.append((blocks.clone(), positions.clone()))
            out.fill_(int(blocks[0, 0]))
            return out

        monkeypatch.setattr(qsa_flydsl, "expand_qsa_block_indices_cuda", expand)
        return fake, expanded

    return install


def test_triton_is_the_default_backend():
    from vllm.config import AttentionConfig

    assert AttentionConfig().qwen4_exp_qsa_backend == "triton"
    assert AttentionConfig(qwen4_exp_qsa_backend="flydsl").qwen4_exp_qsa_backend == (
        "flydsl"
    )
    with pytest.raises(ValueError):
        AttentionConfig(qwen4_exp_qsa_backend="cuda")  # type: ignore[arg-type]


def test_k1_selects_through_the_triton_expand(fake_qsa):
    fake, expanded = fake_qsa()
    inputs = _selection_inputs(5)
    out = qsa_flydsl.flydsl_select_paged_tokens(**inputs)
    assert out is inputs["out"]
    assert fake.k1_calls == [(5, (4,))]
    assert len(expanded) == 1
    assert expanded[0][1].dtype == torch.int64
    assert torch.equal(out, torch.ones_like(out))


def test_k1_chunks_rows_by_the_logits_workspace(fake_qsa, monkeypatch):
    fake, expanded = fake_qsa()
    pages, page_size = 8, 16
    monkeypatch.setattr(
        qsa_flydsl, "_LOGITS_WORKSPACE_BYTES", 2 * pages * page_size * 4
    )
    inputs = _selection_inputs(5, pages, page_size)
    out = qsa_flydsl.flydsl_select_paged_tokens(**inputs)
    assert out is not None
    assert [rows for rows, _ in fake.k1_calls] == [2, 2, 1]
    assert [e[1].tolist() for e in expanded] == [[0, 1], [2, 3], [4]]
    assert out[:, 0].tolist() == [1, 1, 2, 2, 3]


def test_k1_reads_a_strided_indexer_view(fake_qsa):
    fake, _ = fake_qsa()
    inputs = _selection_inputs(3)
    padded = torch.zeros(8, 16, 1, 256, dtype=torch.bfloat16)
    inputs["k_cache"] = padded[..., :128]
    assert not inputs["k_cache"].is_contiguous()
    assert qsa_flydsl.flydsl_select_paged_tokens(**inputs) is inputs["out"]
    assert fake.k1_calls == [(3, (4,))]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda i: i.update(token_topk=1024, out=torch.empty(3, 1027)), id="budget"
        ),
        pytest.param(None, id="unserved"),
    ],
)
def test_k1_falls_back(fake_qsa, mutate):
    fake, _ = fake_qsa(k1_reason=None if mutate else "q must be [M, 4|8, 128]")
    inputs = _selection_inputs(3)
    if mutate:
        mutate(inputs)
    assert qsa_flydsl.flydsl_select_paged_tokens(**inputs) is None
    assert fake.k1_calls == []


def test_k2_writes_output_in_place(fake_qsa):
    fake, _ = fake_qsa()
    inputs = _attention_inputs(3)
    assert not inputs["k_cache"].is_contiguous()
    assert qsa_flydsl.flydsl_sparse_paged_attention(**inputs)
    assert fake.k2_calls == [3]
    assert torch.equal(inputs["out"], torch.ones_like(inputs["out"]))


@pytest.mark.parametrize("case", ["unserved", "strided_out"])
def test_k2_falls_back(fake_qsa, case):
    fake, _ = fake_qsa(k2_reason="unserved" if case == "unserved" else None)
    inputs = _attention_inputs(3)
    if case == "strided_out":
        inputs["out"] = torch.empty(3, 256, 12, dtype=torch.bfloat16).transpose(1, 2)
    assert not qsa_flydsl.flydsl_sparse_paged_attention(**inputs)
    assert fake.k2_calls == []


@pytest.mark.parametrize(
    ("num_prefills", "max_seq_len", "width"),
    [(1, 3000, 2), (1, 1568, 1), (0, 3000, 168)],
)
def test_select_trims_the_table_on_prefill_batches(
    monkeypatch, num_prefills, max_seq_len, width
):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.amd.qsa_indexer import QSAIndexer

    widths = []

    def select(q, k_cache, page_table, *args):
        widths.append(page_table.shape[1])
        return page_table

    monkeypatch.setattr(qsa_flydsl, "flydsl_select_paged_tokens", select)
    indexer = _indexer(True)
    metadata = SimpleNamespace(
        block_table=torch.zeros(2, 168, dtype=torch.int32),
        token_to_req=None,
        logical_positions=None,
        seq_lens=None,
        num_prefills=num_prefills,
        num_decode_tokens=1,
        max_seq_len=max_seq_len,
        storage_block_size=392,
        compress_ratio=COMPRESS_RATIO,
    )
    QSAIndexer._select(indexer, torch.zeros(2, 512), metadata, None)
    assert widths == [width]


def test_select_stays_on_triton_without_the_flydsl_backend(monkeypatch):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.amd import qsa_indexer
    from vllm.models.qwen4_exp.amd.qsa_indexer import QSAIndexer

    def flydsl(*args):
        raise AssertionError("FlyDSL selected without the flydsl backend")

    sentinel = object()
    monkeypatch.setattr(qsa_flydsl, "flydsl_select_paged_tokens", flydsl)
    monkeypatch.setattr(
        "vllm.models.qwen4_exp.amd.ops.qsa.qsa_select_paged_tokens",
        lambda *args, **kwargs: sentinel,
    )
    indexer = _indexer(False)
    metadata = SimpleNamespace(
        block_table=torch.zeros(2, 168, dtype=torch.int32),
        token_to_req=None,
        logical_positions=None,
        seq_lens=None,
        num_prefills=0,
        num_decode_tokens=2,
        max_seq_len=3000,
        storage_block_size=392,
        compress_ratio=COMPRESS_RATIO,
    )
    assert QSAIndexer._select(indexer, None, metadata, None) is sentinel
    assert qsa_indexer  # the module imports without AITER


def test_select_sends_long_prefill_chunks_to_the_mfma_scorer(monkeypatch):
    """Under the flydsl backend a batch's prefill rows of at least
    MFMA_SCORER_MIN_ROWS take the MFMA scorer and its decode rows stay on K1."""
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.amd.ops import qsa
    from vllm.models.qwen4_exp.amd.ops.qsa_mfma_scorer import MFMA_SCORER_MIN_ROWS
    from vllm.models.qwen4_exp.amd.qsa_indexer import QSAIndexer

    calls = []

    def k1(q, k_cache, page_table, token_to_req, positions, seq_lens, topk, r, out):
        calls.append(("k1", q.shape[0], token_to_req[0].item()))
        if out is not None:
            out.fill_(1)
        return page_table

    def scorer(
        q, k_cache, page_table, token_to_req, *args, num_decode_tokens=None, **kwargs
    ):
        calls.append(("mfma", q.shape[0], token_to_req[0].item(), num_decode_tokens))
        args[-1].fill_(2)
        return args[-1]

    monkeypatch.setattr(qsa_flydsl, "flydsl_select_paged_tokens", k1)
    monkeypatch.setattr(qsa, "qsa_select_paged_tokens", scorer)
    decode, prefill = 3, MFMA_SCORER_MIN_ROWS
    rows = decode + prefill
    indexer = _indexer(True, torch.zeros(1, dtype=torch.bfloat16))
    metadata = SimpleNamespace(
        block_table=torch.zeros(4, 168, dtype=torch.int32),
        token_to_req=torch.tensor([0, 1, 2] + [3] * prefill),
        logical_positions=torch.arange(rows),
        seq_lens=None,
        num_prefills=1,
        num_decode_tokens=decode,
        max_seq_len=3000,
        storage_block_size=392,
        compress_ratio=COMPRESS_RATIO,
    )
    q = torch.zeros(rows, 512, dtype=torch.bfloat16)
    out = QSAIndexer._select(indexer, q, metadata, None)
    assert calls == [("mfma", prefill, 3, 0), ("k1", decode, 0)]
    assert out[:decode].eq(1).all() and out[decode:].eq(2).all()

    calls.clear()
    metadata.token_to_req = torch.tensor([0, 1, 2] + [3] * (prefill - 1))
    q = q[: rows - 1]
    QSAIndexer._select(indexer, q, metadata, None)
    assert calls == [("k1", rows - 1, 0)]


def _indexer(use_flydsl, kv_cache=None):
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.amd.qsa_indexer import QSAIndexer

    indexer = SimpleNamespace(
        use_flydsl=use_flydsl,
        compressed_key_cache=SimpleNamespace(kv_cache=kv_cache),
        token_topk=TOKEN_TOPK,
        compress_ratio=COMPRESS_RATIO,
    )
    indexer._select_flydsl = lambda *args: QSAIndexer._select_flydsl(indexer, *args)
    return indexer


def test_missing_aiter_raises(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_flydsl(name, *args, **kwargs):
        if name.startswith("aiter"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    qsa_flydsl.flydsl_qsa.cache_clear()
    monkeypatch.setattr(builtins, "__import__", no_flydsl)
    try:
        with pytest.raises(ImportError, match="aiter"):
            qsa_flydsl.flydsl_qsa()
    finally:
        qsa_flydsl.flydsl_qsa.cache_clear()
