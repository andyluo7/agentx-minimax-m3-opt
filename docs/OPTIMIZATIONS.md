# Optimization summary

## Winning configuration

- TP4. TP2 was 27.2% slower in P90 TPOT; TP8 C1 reduced latency but materially
  reduced throughput per chip.
- FP8 target KV cache and FP8 lightning-indexer KV cache.
- Full lightning-indexer computation. No skip-indexer shortcut is present in
  the validated curve.
- AITER unified attention for both the target model and EAGLE3 drafter.
- EAGLE3 with four proposed tokens and the committed GQA golden synthetic
  acceptance length of 3.02 for performance replay.
- Real-target verification for correctness runs.
- 32K bounded EAGLE3 draft-attention window while retaining the full draft KV
  cache, so old draft KV remains available when sequences revisit context.
- gfx950 3-D unified-attention work partitioned only over the live window.
- AITER score/top-k and sparse paged attention for the lightning indexer.
- Fused QK-norm, RoPE, page-16 K/V, and index-cache insertion.
- Fused input all-reduce plus Gemma RMSNorm.
- Comprehensive startup precompile for indexer, attention, EAGLE metadata,
  drafter GEMMs, and sampler specializations.
- 32K scheduler token budget, `max_num_seqs=256`, CUDA-graph capture size 512,
  and prefix caching.
- Resident KV for C1/C5/C10. Lazy `SimpleCPUOffloadConnector` with 1,499 GiB
  host allocation for C15 through C32.

## Measured contributors

The fused MiniMax-M3 cache-insertion path produced the largest controlled
end-to-end gain:

- median TPOT: 12.9% lower;
- median end-to-end latency: 9.8% lower;
- throughput: 4.8% higher.

The fused input all-reduce plus Gemma RMSNorm path added:

- median TPOT: 4.0% lower;
- median end-to-end latency: 4.7% lower.

The live-window unified-attention kernel produced up to 13.81x B1 and 9.11x
B32 kernel-level speedups at the long-context test shapes. End-to-end gains are
smaller because attention is only one part of the decode step.

## Correctness repair

The original dense ROCm AITER path corrupted output during CUDA-graph replay.
The graph capture builder zeroed `query_start_loc`, while unified attention
consumed that metadata during replay. The invalid path emitted a short valid
prefix followed by NUL bytes and could also appear artificially fast because
required dense attention work was not represented correctly.

The correction preserves the unified-attention query metadata through capture
and replay. It passed target-only smoke, assembled EAGLE3 smoke, the full
GSM8K evaluation, and the matched performance curve.

## Rejected or excluded paths

- Skipping lightning-indexer computation: invalid for the apples-to-apples
  InferenceX result and excluded from the final curve.
- LMCache offload: not the winning offload path for this sweep.
- TP2: worse end-to-end decode latency.
- TP8 C1: lower per-chip throughput despite lower latency.
- ASM sparse attention: disabled in the validated stack.
- Downstream FP32 router GEMM: disabled; the related upstream PR is not needed
  to reproduce these numbers.
- Breakable CUDA graphs: disabled.
- 16K scheduler budget: 2.3% worse matched TPOT and 1.1% lower throughput than
  the retained 32K setting.
- C32 resident: useful as a diagnostic but dominated for the submitted Pareto
  curve and therefore not part of the eight-point winner.
