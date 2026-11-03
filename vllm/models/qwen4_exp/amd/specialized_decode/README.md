# Qwen3.8-Flash-Next specialized decode

`--monokernel` on MI355X/gfx950 runs each pure decode step of 1 to 128 rows
of Qwen3.8-Flash-Next (`amd/Qwen3.8-Flash-Next-Quark-MXFP4`) at TP2 as
persistent FlyDSL launches, in step mode (`runtime/decode.py`). `plan()`
routes each step by its width to one of two kernel families:

| rows | family | launches a layer |
| --- | --- | --- |
| 1-8 | `mono/` (`runtime/mono_launch.py`) | one per block, mailbox hand-offs between a block's stages |
| FULL graph capture sizes in 16-128, multiples of 8 | `wide/` (`runtime/wide_launch.py`) | one per segment, batch-tiled phases behind grid barriers |

Other widths, prefill, mixed and PIECEWISE steps run vLLM's forward. The
embedding, the per-layer embeddings, the final mixer, the head and sampling
stay vLLM's. Both families cut a layer at the same module boundaries:

| unit | mono | wide segment |
| --- | --- | --- |
| attention HC mix (`GatedResidual.mix`) | `front.hc_front` | `attn_mix` |
| GDN: in_proj_qkvz (MXFP4) + in_proj_ba, conv update, gated delta rule, RMSNormGated | `blocks.build_gdn` | `gdn` |
| QSA: qkv_proj and QSA attention | vLLM's (`mono_launch.qsa_core`) | vLLM's, wide indexer and sparse attention (`wide_launch.qsa_core`) |
| out_proj (MXFP4) / gated o_proj (BF16), TP all-reduce, attention HC combine | `blocks.build_oproj` | `attn_out` |
| MLP HC mix | `front.hc_front` | `mlp` |
| router top-10, MXFP4 experts with the fused shared expert (expert 512), TP all-reduce, MLP HC combine | `blocks.build_moe` | `mlp` |

Each family folds the HC combine into its all-reduce epilogue, so the
multi-stream state is written once a unit; the mono blocks also hand the
next front the state's sums of squares.

The wide segments are launched one at a time by default
(`runtime.wide_launch.SPLIT = True`: 180 launches a step). With `SPLIT =
False` they fuse back into one launch per GDN layer and two per QSA layer
(`attn_mix`; `attn_out` + `mlp`), 60 a step. Either way the segments are the
same code (`wide.layer.LayerBuild.segments`, `layer_plan`); a split launch
ends without its last grid barrier, the kernel boundary orders the memory.

## Why two families

Mono streams each block's weights over the whole grid and hands rows between
stages through mailboxes, which is what a few rows need. Wide tiles the batch:
an MoE unit is one expert and up to 16 of its rows, whatever its row count
one CU's stream of the expert's weights (~1.3 MB a rank), so its time floor
does not shrink below 8 rows. Measured on the 1k/8k serving sweeps the mono
kernels win up to 8 rows and the wide kernels from 12 rows (one width
routing threshold, `runtime.MONO_MAX_TOKENS`).

## Wide MoE schedule

Real routing is far from uniform (128 rows of natural text: ~300 distinct
experts a layer, up to ~80 rows on one; uniform: ~470), so the schedule
(`wide/stages/moe.py`, `experts`) adapts to the step's distribution:

- units of one expert are consecutive and run on one XCD (runs of 8 rotate
  over the XCDs), so a hot expert's weights come through one L2;
- a last round at most half full (for example 300 units on 256 CTAs) runs
  as two intermediate halves a unit (half the gate/up rows and the down
  projection's K each, the second half's partial in its own buffer, marked
  for the combine), so the round ends after about half a unit instead of a
  whole one.

## Supported

- TP2 on dedicated, unpartitioned GPUs (256 CUs, one CTA per CU), Model
  Runner V2; eager steps (mono widths) and FULL CUDA graph replays (graph
  padding rows are inert: their GDN state slot is the null block);
- the checkpoint's shapes (`mono/layout.py` and `wide/layout.py`, `SHAPES`)
  and formats: online MXFP4 GDN projections (`--quantization-config.targets`
  for `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`) in the layout of
  whichever AITER FP4 GEMM vLLM picked (ASM: shuffled; Triton: row-major,
  transposed scales), `AITER_MXFP4_MXFP4` experts with
  `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`, a BF16 conv state and an
  FP32 recurrent state with `--no-enable-prefix-caching`.

## Refusals

Default-deny, in three stages, each reporting every reason in one error:

- config, in the model's `__init__` (`runtime/__init__.py`, `unsupported`):
  TP other than 2, PP, DP, EP, EPLB, DCP, PCP, DBO, HC/MoE sequence
  parallelism, LoRA, KV connectors, routed-expert capture, speculative
  decoding (MTP and EAGLE3 included), sleep mode, CPU weight offload, weight
  updates, GDN prefix caching, other cache dtypes, an unfused shared expert,
  the V1 runner, `STOCK_TORCH_COMPILE`, other model shapes, and a platform
  other than unpartitioned gfx950;
- model, after load (`runtime/guards.py`, `model_reasons`): every layer's
  modules, quantization methods, weight layouts and shapes;
- caches, at every KV cache bind (`cache_reasons`): the GDN states' dtypes,
  shapes and slot layout.

The TP ranks agree on every stage, so a rank never refuses alone. Every
kernel of every width is compiled when the executor is built (FlyDSL's
`COMPILE_ONLY`), not on the first step of a width. Without `--monokernel`
nothing here runs.

## Numerics

The stock path's rounding points are kept: bf16 multi-stream state, bf16
gated-residual GEMM outputs, bf16 router logits, MXFP4 MoE activations and
intermediate with AITER's scale rule, bf16 GDN projection outputs and core
output. Mono quantizes the GDN projections' inputs as the AITER GEMM does;
wide runs them W4A16 (the weights decoded to bf16 in registers).

The wide TP all-reduces follow the deployment's QuickReduce setting
(`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION`): with `INT4` (or `INT3`) they send
QuickReduce's INT4 codec (16-column groups, a bf16 scale), 3.2x fewer bytes
on the link; otherwise they are exact bf16 sums in rank order, as stock's
custom all-reduce at decode sizes. Every rank decodes every rank's copy, its
own included, so the ranks agree. The mono all-reduces are exact bf16.

## Failure

A fault in a launch traps the GPU queue: the worker's wait for the step's
output fails and the engine stops, no token is returned from that step.

Every in-kernel wait (mailbox, grid barrier, TP exchange line) gives up after
10 s: the CTA records it in the rank's health words (`mono/abi.py`,
`wide/abi.py`), raises the rank's flag on the output rank and marks the rank
failed, so its later waits exit at once and the step runs to its end.
`FailStop` (`runtime/guards.py`) reads both families' failed words after
every step without blocking and the TP ranks agree on them every 16 steps,
then all raise together and the engine stops.

## Code

- `runtime/`: config refusals and widths (`__init__`), step routing
  (`routing`), model and cache refusals and `FailStop` (`guards`), the
  executor (`decode`), the launches of each family (`mono_launch`,
  `wide_launch`), peer memory (`peer`).
- `mono/`: the front and the blocks (`front`, `blocks`), the shapes
  (`layout`), the health words (`abi`); `stages/`: the GDN, in_proj, router,
  MoE and GEMV stages shared by the blocks; `common/`: the device
  primitives, the tagged mailbox with its bounded waits, the build keys.
- `wide/`: the build key, segments, ABI and scratch layout (`layer`), the
  shapes (`layout`), the health words (`abi`); `common/`: device primitives
  (`ops`), the grid barrier and fold counters (`grid`), build keys (`plan`);
  `stages/`: the batch-tiled GEMM (`tiled`), the gated residual (`hc`), the
  gated delta net (`gdn`), the MoE (`moe`), MXFP4 activations (`mx`), the TP
  exchange (`exchange`).

The mailbox, peer memory, GEMV stage and `FailStop` are copies of the Kimi-K3
decode's (#60683, by hyukjlee, after ROCm/ATOM), trimmed to what these
kernels use, so this package imports nothing of Kimi-K3's.

For profiling, `runtime.wide_launch.TRACE = True` (set in the worker before
the model is built, for example by a `--worker-extension-cls` module) builds
every wide launch with per-phase clock stamps (`Launcher.trace`,
`wide.layer.phase_names`).

## Toolchain

FlyDSL comes in through AITER (`docker/Dockerfile.rocm_base`); startup does
not check its version. `tests/models/qwen4_exp/amd_specialized_decode/`
checks the kernels against vLLM's ops (`test_kernels.py`) and each wide stage
against a float64 reference with the stock rounding points
(`test_stages.py`).
