# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B decode throughput with and without the GEMM + SiluAndMul fusion pass.

    python benchmarks/gemm_silu_mul_ab.py --model PATH_OR_HF_ID [--fuse 0|1]
        [--batches 1 8 32] [--out-tokens 256] [--tp 1]

Run once with --fuse 0 and once with --fuse 1 (separate processes, so that the
compile caches stay apart) and compare the printed tok/s and prompt logprobs.
"""

import argparse
import json
import random
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fuse", type=int, default=1)
    p.add_argument("--batches", nargs="*", type=int, default=[1, 8, 32])
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--out-tokens", type=int, default=256)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--skip-tokenizer-init", action="store_true")
    p.add_argument("--json", help="write results here")
    args = p.parse_args()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        skip_tokenizer_init=args.skip_tokenizer_init,
        compilation_config={"pass_config": {"fuse_gemm_silu_mul": bool(args.fuse)}},
    )
    vocab = llm.llm_engine.model_config.get_vocab_size()
    rng = random.Random(0)

    def prompts(n: int, length: int = args.prompt_len) -> list[TokensPrompt]:
        return [
            TokensPrompt(
                prompt_token_ids=[rng.randrange(100, vocab - 100) for _ in range(length)]
            )
            for _ in range(n)
        ]

    sp = SamplingParams(
        max_tokens=args.out_tokens, temperature=0.0, ignore_eos=True, detokenize=False
    )
    llm.generate(prompts(4), sp, use_tqdm=False)
    res: dict = {"fuse": args.fuse, "model": args.model, "tok_s": {}}
    for b in args.batches:
        ps = prompts(b)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            outs = llm.generate(ps, sp, use_tqdm=False)
            times.append(time.perf_counter() - t0)
        res["tok_s"][b] = b * args.out_tokens / min(times)
        res.setdefault("ids", {})[b] = [list(o.outputs[0].token_ids) for o in outs]
        print(f"RESULT fuse={args.fuse} batch={b}: {res['tok_s'][b]:.1f} tok/s")

    # Teacher-forced prompt logprobs do not depend on earlier sampling, so they
    # compare numerics even where greedy decoding flips near-ties. 4 x 16 tokens
    # stay within the fused (decode-sized) compile range.
    rng.seed(1)
    lp = []
    for o in llm.generate(
        prompts(4, 16),
        SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False),
        use_tqdm=False,
    ):
        lp.append(
            [
                None if d is None else d[t].logprob
                for t, d in zip(o.prompt_token_ids, o.prompt_logprobs)
            ]
        )
    res["prompt_logprobs"] = lp
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
