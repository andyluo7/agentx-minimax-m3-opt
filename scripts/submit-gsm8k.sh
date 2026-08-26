#!/usr/bin/env bash
set -euo pipefail

readonly shared_root="${AGENTX_SHARED_ROOT:-/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx}"
readonly sbatch_script="${AGENTX_SBATCH_SCRIPT:-$shared_root/run-agentx.sbatch}"
readonly recipe="${AGENTX_RECIPE_PATH:-$shared_root/repos/InferenceX/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh}"
readonly node="${MI355X_NODE:-vultr-mi355x-6}"
readonly slurm_account="${SLURM_ACCOUNT:-r7n}"
readonly slurm_partition="${SLURM_PARTITION:-256C8G1H_MI355X_Ubuntu24}"
readonly slurm_reservation="${SLURM_RESERVATION:-aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation}"
readonly run_label="${RUN_LABEL:-minimaxm3-k4-full-indexer-gsm8k-full-$(date -u +%Y%m%dT%H%M%SZ)}"

env \
    -u SYNTHETIC_ACCEPT_LEN \
    -u ALLOW_NON_GOLDEN_ACCEPTANCE \
    -u VLLM_MINIMAX_M3_OVERLAY \
    -u VLLM_EAGLE3_OVERLAY \
    -u VLLM_AITER_UNIFIED_ATTN_OVERLAY \
    -u AITER_INDEXER_OVERLAY \
    -u VLLM_MINIMAX_M3_WARMUP_OVERLAY \
    AGENTX_SHARED_ROOT="$shared_root" \
    AGENTX_RECIPE_OVERRIDE="$recipe" \
    AGENTX_CONTAINER_NAME="${AGENTX_CONTAINER_NAME:-minimaxm3-vllm-rocm-v0271}" \
    TP=4 \
    CONC=1 \
    MAX_NUM_SEQS=256 \
    MAX_NUM_BATCHED_TOKENS=32768 \
    MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
    GPU_MEMORY_UTILIZATION=0.85 \
    KV_OFFLOADING=none \
    TOTAL_CPU_DRAM_GB=1499 \
    NUM_SPEC_TOKENS=4 \
    INDEXER_KV_DTYPE=fp8 \
    KV_CACHE_DTYPE=fp8 \
    AITER_SPARSE_PRECOMPILE=0 \
    AIPERF_EXPERIMENTAL_FAST=0 \
    EVAL_ONLY=true \
    RUN_EVAL=true \
    EVAL_LIMIT=full \
    RUN_LABEL="$run_label" \
    sbatch \
        --parsable \
        --account="$slurm_account" \
        --partition="$slurm_partition" \
        --reservation="$slurm_reservation" \
        --exclusive \
        --gpus-per-node=8 \
        --nodelist="$node" \
        --job-name="${run_label:0:64}" \
        --time=12:00:00 \
        --export=ALL \
        "$sbatch_script"
