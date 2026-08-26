# Validated results

## Full lightning-indexer MI355X curve

The authoritative result set is
`artifacts/performance/full-indexer-curve/validation-report.json`.

| Concurrency | TP | Policy | Throughput tok/s/chip | P90 TPOT ms | Raw P90 tok/s/user | Normalized P90 tok/s/user | Successful measured requests |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 4 | resident | 4,324.51085 | 4.27 | 234.24967 | 110.13732 | 248 |
| 5 | 4 | resident | 7,707.12558 | 5.68 | 175.99448 | 118.63063 | 635 |
| 10 | 4 | resident | 15,685.24936 | 6.26 | 159.79038 | 112.32127 | 1,487 |
| 15 | 4 | vLLM simple offload | 20,764.88786 | 7.46 | 134.11096 | 93.39649 | 2,390 |
| 20 | 4 | vLLM simple offload | 30,078.33753 | 9.20 | 108.70267 | 78.01346 | 2,861 |
| 25 | 4 | vLLM simple offload | 37,459.34595 | 11.13 | 89.82926 | 65.01424 | 3,574 |
| 30 | 4 | vLLM simple offload | 41,315.11798 | 13.73 | 72.81742 | 55.81140 | 3,887 |
| 32 | 4 | vLLM simple offload | 43,683.38852 | 15.59 | 64.13323 | 48.69310 | 3,961 |

All eight points passed these gates:

- full lightning-indexer computation was enabled;
- exactly one hour of measured replay and ten warmup requests per lane;
- zero measured-request errors;
- zero post-profile JIT events;
- zero GPU fault events in the preserved server/wrapper logs;
- `replay_rc=0`;
- expected TP4 resident/offload policy for every concurrency.

C20, C25, and C32 contain respectively one, one, and three aggregate dropped
records including warmup. They still contain zero measured-phase errors. The
validator derives measured accounting from `profile_export.jsonl` and does
not silently equate aggregate warmup drops with measured failures.

## Matched B200 comparison

The B200 reference comes from InferenceX Actions run
[31833401868](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31833401868/attempts/1).

| Point | MI355X tok/s/chip | B200 tok/s/chip | MI355X/B200 |
|---|---:|---:|---:|
| C1 | 4,324.51 | 5,251.61 | 82.3% |
| C5 | 7,707.13 | 8,669.51 | 88.9% |
| C10 | 15,685.25 | 16,428.44 | 95.5% |
| C15 | 20,764.89 | 22,004.10 | 94.4% |
| C20 | 30,078.34 | 32,071.85 | 93.8% |
| C25 | 37,459.35 | 40,482.83 | 92.5% |
| C30 | 41,315.12 | 42,239.94 | 97.8% |
| C32 | 43,683.39 | 44,143.90 | 99.0% |

The public pre-optimization MI355X reference comes from InferenceX Actions run
[31558297538](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31558297538/attempts/1).
The plot script resolves the exact points from the InferenceX API and pins the
two run IDs when regenerating the chart.

## Correctness

Artifact: `artifacts/correctness/gsm8k-full-k4/`

- 1,319 original samples and 1,319 effective samples
- strict exact match: `0.9689158453373768`
- flexible extraction: `0.9681576952236542`
- real target verification; no synthetic acceptance in the server config
- MiniMax-M3 reasoning parser enabled

Performance and correctness are deliberately separate. The performance replay
uses the committed MiniMax-M3 EAGLE3-GQA golden synthetic acceptance length of
3.02 for four proposals. The GSM8K run removes synthetic rejection sampling.

## Repaired steady-state profile

Artifact: `artifacts/profile/trace-summary.md`

The repaired 262K-context profile ranked the remaining GPU time as:

| Category | Share of profiled GPU time |
|---|---:|
| Tiny-M MoE | 21.7% |
| Dense GEMV | 16.2% |
| TP4 collectives | 15.0% |
| Dense attention | 8.7% |
| Sparse attention | 7.2% |
| Index scoring | 2.7% |

Dense full-context attention ceased to be the dominant bottleneck after the
CUDA-graph metadata correction. The remaining gap is primarily repeated
per-token drafter, tiny-M, launch, and TP synchronization cost.
