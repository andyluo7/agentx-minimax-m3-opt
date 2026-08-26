# Upstream PR status and dependency analysis

Status snapshot: 2026-08-26, refreshed directly from GitHub.

`Draft / Open` means the GitHub lifecycle state is open and the PR is marked
draft. GitHub's merge-readiness value `BLOCKED` does not mean the PR is closed;
it generally reflects draft/review/check requirements.

## Focused PRs opened for this work

| PR | Change | Head | Lifecycle status | Dependency |
|---|---|---|---|---|
| [vLLM #53821](https://github.com/vllm-project/vllm/pull/53821) | Preserve ROCm unified-attention query metadata during CUDA-graph replay | `17cccfedac8a` | **Draft / Open** | Independent correctness prerequisite |
| [vLLM #53827](https://github.com/vllm-project/vllm/pull/53827) | Bounded Llama EAGLE3 draft attention with full draft-KV retention | `641376f8f069` | **Draft / Open** | Pairs with AITER #5004 for intended MI355X efficiency |
| [AITER #5004](https://github.com/ROCm/aiter/pull/5004) | gfx950 live-window 3-D unified-attention optimization | `adcda0d3a8a6` | **Draft / Open** | Based on merged AITER #4918; performance companion to vLLM #53827 |
| [vLLM #53833](https://github.com/vllm-project/vllm/pull/53833) | Integrate AITER fused MiniMax-M3 cache insertion | `2cb93a03ebf8` | **Draft / Open** | Requires AITER #4813; rebase after vLLM #52664 |

## Required or reused upstream work

| PR | Role | Head | Lifecycle status | Relationship |
|---|---|---|---|---|
| [AITER #4787](https://github.com/ROCm/aiter/pull/4787) | Lightning-indexer score/top-k kernels | `cb3c7a628645` | **Open** | Required by vLLM #52664 |
| [vLLM #52849](https://github.com/vllm-project/vllm/pull/52849) | AITER sparse paged attention for MiniMax-M3 MTP/dense layers | `148cce51cd75` | **Open**, approved | vLLM #52664 must be rebased onto it/current main |
| [vLLM #52664](https://github.com/vllm-project/vllm/pull/52664) | AITER lightning-indexer integration | `92b66b2bdf03` | **Draft / Open** | Requires AITER #4787 and vLLM #52849 |
| [AITER #4813](https://github.com/ROCm/aiter/pull/4813) | Fused QK-norm, RoPE, and cache-insert kernel | `266922c417ce` | **Draft / Open**, conflicts | Required by vLLM #53833; clean helper rebase is `andyluo7/aiter:rebase/minimaxm3-fused-cache-insert` at `08587c2c0` |
| [vLLM #53695](https://github.com/vllm-project/vllm/pull/53695) | KV-connector support for ROCm AITER unified attention | `9e7aa17a16eb` | **Open** | Required for C15-C32 `SimpleCPUOffloadConnector` points |
| [vLLM #47270](https://github.com/vllm-project/vllm/pull/47270) | Fused input all-reduce plus Gemma RMSNorm | `f53708556e0f` | **Draft / Open** | Independent performance component |
| [vLLM #49170](https://github.com/vllm-project/vllm/pull/49170) | Generic AITER unified-attention warmup | `2de511e91424` | **Open**, conflicts | Reuse after rebase; sparse-PA warmup still needs recipe coverage |
| [AITER #4918](https://github.com/ROCm/aiter/pull/4918) | gfx950 unified-attention selector tuning | `c76ca13f3625` | **Merged** | Included in AITER #5004's base |

## Dependency flow

```text
AITER #4787 -----------------+
                              +--> vLLM #52664 --+
vLLM #52849 -----------------+                   +--> vLLM #53833
                                                  |
AITER #4813 -------------------------------------+

AITER #4918 (merged) --> AITER #5004
                               +
                         vLLM #53827

vLLM #53695 --> native vLLM simple KV offload support
vLLM #53821 --> CUDA-graph correctness
vLLM #47270 --> independent collective/norm performance
vLLM #49170 --> generic unified-attention warmup
```

## Landing order

1. Land AITER #4787 and vLLM #52849.
2. Rebase vLLM #52664 onto merged #52849/current main, fix DCO, and validate
   the exact new head on MI355X.
3. Rebase and land AITER #4813 using the prepared helper branch if accepted by
   its author.
4. Rebase vLLM #53833 after #52664; they overlap in
   `vllm/models/minimax_m3/amd/model.py`.
5. Land vLLM #53695 for native offload compatibility.
6. Land vLLM #53821 independently.
7. Land vLLM #53827 with AITER #5004.
8. Rebase/reuse vLLM #49170 and retain only the missing sparse-PA warmup in the
   workload-owned helper.
9. Reuse vLLM #47270 independently.

## Alternatives that are not dependencies

- [vLLM #53448](https://github.com/vllm-project/vllm/pull/53448) is a draft
  Triton lightning-indexer alternative. It conflicts functionally with #52664
  and should not be stacked on this result.
- [vLLM #52668](https://github.com/vllm-project/vllm/pull/52668) is a draft
  BF16x3 router GEMM. The validated recipe disables that downstream path, so it
  is not required.

The original [InferenceX #2726](https://github.com/SemiAnalysisAI/InferenceX/pull/2726)
is **Closed**, not merged. A patch-free replacement should be submitted only
after an official image contains the upstream stack and passes full GSM8K, C1,
and C32-offload validation.

Run `scripts/refresh-pr-status.sh` to refresh lifecycle and merge-readiness
states without changing this analysis automatically.
