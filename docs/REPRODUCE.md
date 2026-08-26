# How to reproduce the MiniMax-M3 MI355X AgentX results

This document reproduces the preserved full-lightning-indexer result. It does
not reproduce the earlier skip-indexer diagnostic curve.

## 1. Required hardware and access

- One or more MI355X nodes with eight GPUs per node.
- TP4 is used by the model. On AAC17, request the complete eight-GPU node to
  preserve exclusivity and match the validated scheduler contract.
- Approximately 1.5 TiB of usable host DRAM for C15-C32 vLLM simple KV
  offload. The recipe requests 1,499 GiB across the four TP ranks.
- Shared storage for models, the InferenceX checkout, results, and Enroot
  assets; node-local `/tmp` for runtime and JIT caches.
- Hugging Face access to `amd/MiniMax-M3-MXFP4` and
  `Inferact/MiniMax-M3-EAGLE3-GQA`.
- Access to the InferenceX AgentX dataset used by AIPerf:
  `semianalysis_cc_traces_weka_062126`.

AAC17 launch defaults in the scripts are:

```text
account:     r7n
partition:   256C8G1H_MI355X_Ubuntu24
reservation: aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
GPUs/node:   8
```

Override `SLURM_ACCOUNT`, `SLURM_PARTITION`, and `SLURM_RESERVATION` when using
another cluster.

## 2. Verify the evidence bundle

```bash
git clone git@github.com:andyluo7/agentx-minimax-m3-opt.git
cd agentx-minimax-m3-opt
sha256sum -c SHA256SUMS
```

On macOS, use `shasum -a 256 -c SHA256SUMS`.

## 3. Pin the software and models

Validated container:

```text
vllm/vllm-openai-rocm:v0.27.1
sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7
```

Validated model revisions:

```text
amd/MiniMax-M3-MXFP4
b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a

Inferact/MiniMax-M3-EAGLE3-GQA
96692486b5fd38ebf8fd2a5f6bb53427d30819a8
```

The recipe downloads missing checkpoints with `hf download`. For a controlled
run, pre-stage both revisions and verify the safetensor index references every
local shard.

Check the image digest before importing it into Enroot:

```bash
docker pull vllm/vllm-openai-rocm:v0.27.1
docker inspect --format '{{json .RepoDigests}}' \
  vllm/vllm-openai-rocm:v0.27.1
```

Do not silently substitute a newer tag. A newer official image belongs to the
patch-free validation path described at the end of this document.

## 4. Prepare the InferenceX checkout

The validated curve recorded InferenceX commit:

```text
62bf882f2df0d732752bc9d83caa3ee2324bda79
```

Create the checkout and place the self-contained historical recipe bundle into
the expected directory:

```bash
git clone https://github.com/SemiAnalysisAI/InferenceX.git
git -C InferenceX checkout 62bf882f2df0d732752bc9d83caa3ee2324bda79

cp -R recipes/inferencex/. \
  InferenceX/benchmarks/single_node/agentic/
cp source/inferencex/eval/lm_eval_sitecustomize.py \
  InferenceX/utils/evals/patches/lm_eval_sitecustomize.py
cp source/inferencex/golden/minimaxm3_eagle3_gqa.yaml \
  InferenceX/golden_al_distribution/minimaxm3_eagle3_gqa.yaml
```

The recipe applies the archived vLLM/AITER delta only inside the ephemeral,
writable benchmark container. Patch application is checksum-gated and refuses
an unexpected or partially patched base.

The archived patch mechanism is for historical reproduction only. It was the
reason InferenceX #2726 was closed and must not be used in a replacement
InferenceX PR.

## 5. Stage the AAC17 runner

Choose a shared directory visible on every selected node:

```bash
export AGENTX_SHARED_ROOT=/shared/data/R7N/$USER/minimaxm3-agentx
mkdir -p \
  "$AGENTX_SHARED_ROOT/repos" \
  "$AGENTX_SHARED_ROOT/results" \
  "$AGENTX_SHARED_ROOT/logs" \
  "$AGENTX_SHARED_ROOT/storage/models" \
  "$AGENTX_SHARED_ROOT/storage/hf"

rsync -a --delete InferenceX/ "$AGENTX_SHARED_ROOT/repos/InferenceX/"
install -m 0755 scripts/run-agentx.sh "$AGENTX_SHARED_ROOT/run-agentx.sh"
install -m 0755 scripts/container-run-agentx.sh \
  "$AGENTX_SHARED_ROOT/container-run-agentx.sh"
install -m 0755 scripts/submit-full-indexer-complete-curve.sh \
  "$AGENTX_SHARED_ROOT/submit-full-indexer-complete-curve.sh"
install -m 0755 scripts/submit-gsm8k.sh \
  "$AGENTX_SHARED_ROOT/submit-gsm8k.sh"
install -m 0644 scripts/slurm/run-agentx.sbatch \
  "$AGENTX_SHARED_ROOT/run-agentx.sbatch"
```

The runner expects an Enroot command file at
`$AGENTX_SHARED_ROOT/enroot-command.rc` and an imported container named
`minimaxm3-vllm-rocm-v0271`. On nodes where Pyxis prefixes imported names, set
`AGENTX_CONTAINER_NAME=pyxis_minimaxm3-vllm-rocm-v0271`.

## 6. Preflight every node

Do not treat `sinfo IDLE` as sufficient evidence. Allocate the node exclusively
and check GPU use, VRAM, KFD users, containers, processes, memory, disk, and
recent GPU faults before launching a performance point.

Example AAC17 allocation probe:

```bash
srun \
  --account=r7n \
  --partition=256C8G1H_MI355X_Ubuntu24 \
  --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation \
  --nodes=1 \
  --ntasks=1 \
  --exclusive \
  --gpus-per-node=8 \
  --nodelist=vultr-mi355x-4 \
  --time=00:05:00 \
  bash -lc 'hostname; rocm-smi --showuse --showmemuse --showpids; fuser /dev/kfd || true; who; enroot list; uptime; free -h; df -h / /tmp /shared/data; journalctl -k --since "2 hours ago" --no-pager | grep -Ei "amdgpu|xgmi|ras|gpu reset|page fault" | tail -80 || true'
```

Do not stop another user's process or container. Select another node if the
allocation is not exclusive and clean.

## 7. Run the complete eight-point sweep

The default mapping uses three nodes and serializes dependent points on each:

```text
node A: C15 offload -> C30 offload -> C1 resident
node B: C20 offload -> C32 offload -> C5 resident
node C: C25 offload -> C10 resident
```

Launch:

```bash
export AGENTX_SHARED_ROOT=/shared/data/R7N/$USER/minimaxm3-agentx
export MI355X_NODE_A=vultr-mi355x-4
export MI355X_NODE_B=vultr-mi355x-5
export MI355X_NODE_C=vultr-mi355x-6
export AGENTX_CONTAINER_NAME=pyxis_minimaxm3-vllm-rocm-v0271

bash "$AGENTX_SHARED_ROOT/submit-full-indexer-complete-curve.sh"
```

For a single available node, set all three node variables to the same node.
Slurm exclusivity will serialize the jobs, although the total wall-clock time
will be much longer.

The effective policy is fixed by the launcher:

- TP4, `max_num_seqs=256`, 32K batch-token budget;
- capture size 512, GPU memory utilization 0.85;
- FP8 target and indexer KV;
- AITER target/draft attention;
- EAGLE3 k=4 with 32K draft window;
- resident C1/C5/C10;
- lazy `SimpleCPUOffloadConnector`, 1,499 GiB, for C15-C32;
- 3,600 measured seconds and ten warmup requests per lane;
- full lightning indexer, no index-cache skip override.

Each run writes aggregate JSON, raw AIPerf records, commands, metadata, and logs
under `$AGENTX_SHARED_ROOT/results/$RUN_LABEL`.

## 8. Run full GSM8K correctness

Performance replay uses synthetic acceptance. Run correctness independently:

```bash
export AGENTX_SHARED_ROOT=/shared/data/R7N/$USER/minimaxm3-agentx
export MI355X_NODE=vultr-mi355x-6
export AGENTX_CONTAINER_NAME=minimaxm3-vllm-rocm-v0271

bash "$AGENTX_SHARED_ROOT/submit-gsm8k.sh"
```

The recipe removes synthetic rejection sampling when `EVAL_ONLY=true` and runs
the complete 1,319-sample GSM8K task with the MiniMax-M3 reasoning parser.
Require both strict and flexible exact-match scores to remain at or above the
InferenceX 0.90 gate.

## 9. Collect and validate

Copy the eight result directories into one local artifact root. Preserve at
least these files for every point:

```text
aggregate JSON
aiperf_artifacts/profile_export.jsonl
benchmark_command.txt
vllm_command.txt
run-metadata.txt
console.log
server.log
```

Run the validator:

```bash
python3 scripts/validation/validate_full_indexer_complete_curve.py \
  artifacts/performance/full-indexer-curve \
  --output /tmp/minimaxm3-validation-report.json
jq '.' /tmp/minimaxm3-validation-report.json
```

Promotion requires:

- all eight expected points and policies;
- `overall_pass=true`;
- zero measured errors and GPU faults;
- zero measured-phase JIT events;
- `replay_rc=0`;
- a complete real-target GSM8K run.

## 10. Regenerate the Pareto chart

The chart script fetches only the pinned public MI355X and B200 Actions runs
and combines them with the local validated curve:

```bash
MPLCONFIGDIR=/tmp/mpl-minimaxm3 \
python3 scripts/validation/plot_agentx_pareto.py \
  --artifact-root artifacts/performance/full-indexer-curve \
  --output-prefix artifacts/plots/minimax-m3-agentx-full-indexer
```

This creates PNG, PDF, and CSV outputs. The x-axis is normalized P90
interactivity and the y-axis is throughput per chip; higher is better in both
directions.

## 11. Patch-free upstream validation

The historical patch bundle must eventually be replaced by an official image
containing the PR stack in `UPSTREAM_PRS.md`. Before opening a replacement
InferenceX PR:

1. record the official image tag and digest;
2. run the exact current vLLM #52664 head rather than relying on the older
   integration commit used by the original sweep;
3. verify zero measured JIT without source overlays;
4. rerun a targeted smoke, all 1,319 GSM8K samples, C1, and C32 offload;
5. submit only the patch-free InferenceX recipe and workload-owned warmup.
