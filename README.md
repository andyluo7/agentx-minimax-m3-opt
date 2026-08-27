# MiniMax-M3 AgentX optimization on MI355X

This repository is the reproducibility and evidence bundle for the MiniMax-M3
FP4 AgentX optimization on AMD Instinct MI355X. It contains the complete
validated eight-point MI355X curve, the full 1,319-sample GSM8K correctness
run, the exact benchmark commands and runtime metadata, the InferenceX recipe
bundle used before upstreaming, validation and plotting scripts, and the
vLLM/AITER upstream PR dependency analysis.

The preserved result uses the full lightning-indexer computation. It does not
use the earlier skip-indexer diagnostic shortcut.

## Validated curve

All performance points use TP4, FP8 target KV, FP8 lightning-indexer KV,
EAGLE3 with four proposals, a 32K bounded draft-attention window with full
draft-KV retention, AITER target and draft attention, prefix caching, a 32K
scheduler budget, and a one-hour measured AgentX profile.

| Point | KV policy | Throughput (tok/s/chip) | P90 TPOT | P90 normalized interactivity |
|---|---|---:|---:|---:|
| C1 | resident | 4,324.51 | 4.27 ms | 110.14 |
| C5 | resident | 7,707.13 | 5.68 ms | 118.63 |
| C10 | resident | 15,685.25 | 6.26 ms | 112.32 |
| C15 | vLLM simple DRAM offload | 20,764.89 | 7.46 ms | 93.40 |
| C20 | vLLM simple DRAM offload | 30,078.34 | 9.20 ms | 78.01 |
| C25 | vLLM simple DRAM offload | 37,459.35 | 11.13 ms | 65.01 |
| C30 | vLLM simple DRAM offload | 41,315.12 | 13.73 ms | 55.81 |
| C32 | vLLM simple DRAM offload | 43,683.39 | 15.59 ms | 48.69 |

The C32 result reaches 99.0% of the matched B200 throughput. C1 reaches 82.3%
of B200 throughput and 69.96% of B200 raw P90 interactivity. Every point has
zero measured-request errors, zero GPU faults, zero measured-phase JIT events,
and a successful replay exit code. Some high-concurrency aggregates contain
warmup-only dropped records; the validation report keeps warmup accounting
separate from the measured interval.

Correctness was evaluated separately with real target verification rather
than synthetic acceptance. The preserved 1,319/1,319 GSM8K run scored
0.968916 strict exact match and 0.968158 flexible extraction.

## Repository contents

- `artifacts/performance/full-indexer-curve/`: raw aggregate JSON, request-level
  AIPerf exports, server and wrapper logs, exact commands, run metadata, and the
  machine-readable eight-point validation report.
- `artifacts/correctness/gsm8k-full-k4/`: full GSM8K result, all sample records,
  exact server command, logs, and environment metadata.
- `artifacts/profile/`: repaired long-context trace summary and profile command.
- `artifacts/current-head/pr52664/`: exact-head focused test evidence and the
  in-progress patch-free migration validation record.
- `artifacts/plots/`: regenerated MI355X versus public MI355X/B200 Pareto chart.
- `recipes/inferencex/`: self-contained historical InferenceX recipe bundle for
  the pinned vLLM v0.27.1 ROCm image. It includes the archived runtime patches
  because that is how the validated result was produced.
- `scripts/`: AAC17/Slurm launchers, validators, profilers, smoke tests, and PR
  status refresh tooling.
- `docs/REPRODUCE.md`: end-to-end setup, full sweep, GSM8K, validation, and plot
  regeneration instructions.
- `docs/UPSTREAM_PRS.md`: live status snapshot and dependency/landing order for
  the focused vLLM and AITER PRs replacing the archived patches.
- `docs/PR_52664_REVIEW_PACKET.md`: exact AITER gates, focused tests,
  non-duplication analysis, DCO blockers, and current-head MI355X validation
  requirements for vLLM #52664.
- `docs/PR_52664_PROPOSED_BODY.md`: concise, community-compliant PR description
  for the human submitter to review and post after the final gates pass.
- `docs/OPTIMIZATIONS.md`: explanation of the winning stack and rejected paths.
- `SHA256SUMS`: integrity manifest for every tracked evidence file.

## Reproduction paths

There are two distinct paths:

1. **Historical result reproduction:** use the pinned image and the
   self-contained recipe/patch bundle in this repository. This is the only
   path expected to reproduce the preserved August 2026 artifacts today.
2. **Patch-free upstream reproduction:** wait for the PR stack in
   `docs/UPSTREAM_PRS.md` to land in an official vLLM/AITER image, then rerun
   GSM8K, C1, and C32 before submitting a new InferenceX recipe.

InferenceX PR #2726 was closed because source patches are not accepted as a
first-class recipe experience. The archived patches here are provenance and
reproduction material, not a proposal to resubmit a patched InferenceX PR.

Start with [docs/REPRODUCE.md](docs/REPRODUCE.md) and verify the bundle with:

```bash
sha256sum -c SHA256SUMS
python3 scripts/validation/validate_full_indexer_complete_curve.py \
  artifacts/performance/full-indexer-curve \
  --output /tmp/minimaxm3-validation-report.json
jq '.overall_pass' /tmp/minimaxm3-validation-report.json
```

Expected output is `true`.

## Claim boundary

Synthetic EAGLE acceptance length 3.02 is used only for standardized AgentX
performance replay. It is not correctness evidence. The GSM8K artifact uses
real target verification. No model weights, Hugging Face credentials, SSH
keys, GitHub tokens, or private AgentX trace payloads are included.
