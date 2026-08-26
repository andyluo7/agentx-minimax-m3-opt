# MiniMax-M3 MI355X upstreaming status

Snapshot: 2026-08-25 America/Los_Angeles

The replacement for closed InferenceX PR #2726 is a set of focused upstream
changes. The InferenceX recipe should not be resubmitted until these changes are
available in an official vLLM/AITER image and the patch-free image has passed
the real-indexer AgentX validation curve.

The upstream source split is complete, but the patch-free recipe migration is
not complete yet. The current standalone precompile helper covers the AITER
indexer, EAGLE metadata, draft GEMMs, and sampler kernels; the old in-vLLM
warmup patch also covered unified-attention and sparse-PA specializations.
Existing vLLM #49170 provides a generic AITER unified-attention warmup, but it
currently conflicts with `main` and needs a rebase plus exact-stack validation.
Reuse it rather than submitting a duplicate. Keep the remaining MiniMax-M3
sparse-PA coverage in an InferenceX-owned helper (or use an equivalent
unmeasured warmup workload), then verify that no JIT compilation occurs during
the measured interval.

## New focused PRs

| Concern | Upstream PR | Head | State | Evidence |
| --- | --- | --- | --- | --- |
| Preserve ROCm unified-attention query metadata during CUDA-graph replay | vLLM #53821 | `17cccfedac8a` | Draft, mergeable, DCO clean; CI authorization gate | 1,319/1,319 GSM8K requests, strict EM 0.969674 on the integrated MI355X stack |
| Bound Llama EAGLE3 draft attention while retaining full draft KV | vLLM #53827 | `641376f8f069` | Draft, mergeable, DCO clean; CI authorization gate | Focused config/cache-spec tests plus long-context AgentX A/B |
| Partition gfx950 3-D unified attention over the live sliding window | AITER #5004 | `adcda0d3a8a6` | Draft, mergeable; Black/Ruff pass | BF16/FP8 query and shuffled/unshuffled FP8 KV correctness; up to 13.81x B1 and 9.11x B32 kernel speedup at 262K/32K |
| Integrate AITER fused MiniMax-M3 QK-norm/RoPE/cache insertion | vLLM #53833 | `2cb93a03ebf8` | Draft, mergeable, DCO clean; CI authorization gate | Full/skip index-branch GPU smoke; 1,319-sample GSM8K integration; controlled C1 TPOT -12.9% |

## Existing upstream work to reuse

| Concern | Existing PR | Current disposition |
| --- | --- | --- |
| AITER fused cache-insert kernel | AITER #4813 | Do not duplicate. It conflicts with current main. Clean rebase helper: `andyluo7/aiter:rebase/minimaxm3-fused-cache-insert` at `08587c2c0`; it keeps main's `composable_kernel` pointer and has no unrelated submodule delta. |
| AITER sparse PA for MiniMax-M3 MTP/dense paths | vLLM #52849 | Approved and AMD CI passed; DCO still requires author action. |
| KV-connector support for ROCm AITER unified attention | vLLM #53695 | Reuse; ready, DCO and pre-commit pass. The exact four-line override is required for the C15-C32 `SimpleCPUOffloadConnector` points. Added sustained MI355X C32 validation evidence in PR comment `5420746608`. |
| Generic AITER unified-attention warmup | vLLM #49170 | Reuse after rebase; currently conflicts with `main`. Its model-derived signature sweep can replace the unified-attention portion of the downstream warmup, but it does not cover MiniMax-M3 sparse PA. |
| AITER lightning-indexer integration | vLLM #52664 | Draft, stacked on the pre-merge-update #52849 head, and DCO requires author action. Depends on AITER #4787. |
| AITER score/top-k kernels | AITER #4787 | Open and mergeable; current CI is green. |
| Triton lightning-indexer optimization | vLLM #53448 | Alternative/fallback implementation, not a drop-in addition to #52664; the branches conflict in three files and its pre-commit currently fails. |
| Input all-reduce plus Gemma RMSNorm fusion | vLLM #47270 | Reuse; it merges cleanly with #52664 and #53833. |
| BF16x3 router GEMM | vLLM #52668 | Existing PR, but the final InferenceX recipe explicitly disables the downstream FP32-router path, so it is not required for reproducing the submitted result. |
| gfx950 unified-attention selector tuning | AITER #4918 | Merged; included in current AITER main and in #5004's base. |

## Dependency and landing order

1. Land AITER #4787 and vLLM #52849.
2. Rebase vLLM #52664 onto the merged #52849/current main, obtain DCO, and
   validate its exact new head with AITER #4787.
3. Rebase and land AITER #4813 using the prepared helper branch if its author
   wants it.
4. Rebase vLLM #53833 after #52664. The two are functionally complementary but
   currently have a content conflict in `vllm/models/minimax_m3/amd/model.py`.
5. Land vLLM #53695 for the block-first unified-attention KV-connector
   capability used by `SimpleCPUOffloadConnector`.
6. Land vLLM #53821 independently; it is the correctness prerequisite for
   `ROCM_AITER_UNIFIED_ATTN` with CUDA graphs.
7. Land vLLM #53827 and AITER #5004 together for the bounded-draft-attention
   configuration. #53827 is correct without #5004, but #5004 supplies the
   intended gfx950 kernel efficiency.
8. Rebase/reuse vLLM #49170 for generic AITER unified-attention warmup; retain
   only the still-missing sparse-PA warmup in the InferenceX recipe.
9. Reuse vLLM #47270 independently. Do not require #52668 for the reproduction.

## Original patch disposition

| InferenceX patch area | Destination |
| --- | --- |
| `rocm_aiter_unified_attn.py` CUDA-graph metadata repair | vLLM #53821 |
| `rocm_aiter_unified_attn.py` KV-connector capability | Existing vLLM #53695 |
| `speculative.py`, generic attention cache spec, `llama.py`, `llama_eagle3.py` | vLLM #53827 |
| AITER unified-attention live-window 3-D changes | AITER #5004 |
| AITER fused QK-norm/RoPE/cache-insert sources | Existing AITER #4813 |
| vLLM fused cache-insert dispatch/fallback | vLLM #53833 |
| AITER indexer and sparse-PA integration | Existing AITER #4787, vLLM #52849, and vLLM #52664 |
| Input all-reduce/Gemma RMSNorm fusion | Existing vLLM #47270 |
| FP32 router kernel | Existing vLLM #52668; not used by the final recipe |
| Large dynamic-kernel precompile/warmup | Reuse vLLM #49170 for generic unified-attention warmup after rebase. Keep AgentX shape coverage and the remaining sparse-PA warmup as InferenceX recipe/helper logic; do not submit the 600-line model/workload-specific warmup as a vLLM source patch. |

## Intentionally omitted downstream controls

The original v0.27.1 patch also carried experiment-only environment switches
for choosing the AITER versus vLLM unified-attention kernel, query/output
quantization, cache writer, ASM sparse attention, and the FP32 router GEMM.
They are not required by the winning configuration and should not be upstreamed
as part of this result:

- the winning unified-attention choices are already the upstream defaults;
- AITER #5004 replaces the forced 3-D switch with a gfx950 dispatch heuristic;
- AITER #4813 plus vLLM #53833 replace the cache-insert environment gate with
  capability detection and an older-AITER fallback;
- ASM sparse attention and the downstream FP32-router path were disabled in the
  final recipe;
- the duplicate `csrc/fused_include` files existed only to patch an installed
  wheel/source-overlay layout and do not belong in AITER upstream.

## Validation caveat

The preserved full-indexer sweep validates the downstream integration adapted
from vLLM #52664 commit `e41bdb0d7595`. The current #52664 head is
`92b66b2bdf03` and contains material changes for the newer KV-cache layout and
AITER module path. It must receive an exact-head MI355X smoke, full GSM8K, and
at least C1/C32 performance validation before the patch-free InferenceX recipe
can claim the existing results.

The three new vLLM PRs currently show failed `pre-run-check` jobs only because
the author has zero merged vLLM PRs and the drafts do not yet have a `ready` or
`verified` label. This is an authorization gate, not a test failure.
