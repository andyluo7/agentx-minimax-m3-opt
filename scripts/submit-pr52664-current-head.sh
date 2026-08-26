#!/usr/bin/env bash
set -euo pipefail

readonly shared_root="${AGENTX_SHARED_ROOT:-/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx}"
readonly validation_root="$shared_root/upstream-validation"
readonly sbatch_script="${AGENTX_SBATCH_SCRIPT:-$shared_root/run-agentx-current-head.sbatch}"
readonly runner="${AGENTX_RUNNER_PATH:-$shared_root/run-agentx-current-head.sh}"
readonly recipe="${AGENTX_RECIPE_PATH:-$shared_root/repos/InferenceX/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp_current_head.sh}"
readonly node="${MI355X_NODE:-vultr-mi355x-1}"
readonly slurm_account="${SLURM_ACCOUNT:-r7n}"
readonly slurm_partition="${SLURM_PARTITION:-256C8G1H_MI355X_Ubuntu24}"
readonly slurm_reservation="${SLURM_RESERVATION:-aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation}"

readonly pr_sha=ee5f001be6454bc9616dcf0db9e4276efe1387c6
readonly integration_sha=720d05565afb12f9a812607b1618a18973db11bc
readonly aiter_sha=cb3c7a628645dd9b03610d36b543d39225e2cde5
readonly vllm_root="$validation_root/vllm-${integration_sha:0:12}"
readonly aiter_root="$validation_root/aiter-${aiter_sha:0:12}"
readonly dependency_heads="vllm#52849=78e1f096add72ec2816eab5da08cba221260142b,vllm#53695=9e7aa17a16eb1435aa2ff8c40473498b905b310e,vllm#53821=251d3d14d778b8b86ec9025f44610dd809d97768,aiter#4787=$aiter_sha"

submit_run() {
    local label="$1"
    local dependency="$2"
    local time_limit="$3"
    local concurrency="$4"
    local eval_only="$5"
    local eval_limit="$6"
    local kv_offloading="$7"
    local kv_backend="$8"
    local dependency_args=()

    if [[ -n "$dependency" ]]; then
        dependency_args+=(--dependency="afterok:$dependency")
    fi

    env \
        -u SYNTHETIC_ACCEPT_LEN \
        -u ALLOW_NON_GOLDEN_ACCEPTANCE \
        -u AITER_INDEXER_OVERLAY \
        -u VLLM_EAGLE3_OVERLAY \
        -u VLLM_MINIMAX_M3_WARMUP_OVERLAY \
        AGENTX_SHARED_ROOT="$shared_root" \
        AGENTX_RUNNER_PATH="$runner" \
        AGENTX_RECIPE_OVERRIDE="$recipe" \
        AGENTX_CONTAINER_NAME="${AGENTX_CONTAINER_NAME:-minimaxm3-vllm-rocm-v0271}" \
        VLLM_MINIMAX_M3_OVERLAY="$vllm_root/vllm/models/minimax_m3" \
        VLLM_ROCM_AITER_FA_OVERLAY="$vllm_root/vllm/v1/attention/backends" \
        VLLM_AITER_UNIFIED_ATTN_OVERLAY="$vllm_root/vllm/v1/attention/backends" \
        PYTHONPATH="$aiter_root" \
        AITER_META_DIR="$aiter_root" \
        VLLM_EXACT_SOURCE_SHA="$pr_sha" \
        VLLM_INTEGRATION_SOURCE_SHA="$integration_sha" \
        VLLM_DEPENDENCY_HEADS="$dependency_heads" \
        AITER_EXACT_SOURCE_SHA="$aiter_sha" \
        MINIMAX_M3_APPLY_ARCHIVED_PATCHES=0 \
        VLLM_MINIMAX_M3_FUSED_CACHE_INSERT=0 \
        VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA=0 \
        VLLM_MINIMAX_M3_ROCM_FP32_ROUTER_GEMM=0 \
        VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP=0 \
        VLLM_MINIMAX_M3_ASM_SPARSE_ATTN=0 \
        AITER_UNIFIED_ATTN_SLIDING_DECODE_3D=0 \
        DRAFT_ATTENTION_WINDOW=none \
        TARGET_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \
        DRAFT_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \
        INDEXER_KV_DTYPE=fp8 \
        KV_CACHE_DTYPE=fp8 \
        TP=4 \
        CONC="$concurrency" \
        MAX_NUM_SEQS=256 \
        MAX_NUM_BATCHED_TOKENS=32768 \
        MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
        GPU_MEMORY_UTILIZATION=0.85 \
        KV_OFFLOADING="$kv_offloading" \
        KV_OFFLOAD_BACKEND="$kv_backend" \
        TOTAL_CPU_DRAM_GB=1499 \
        NUM_SPEC_TOKENS=4 \
        AITER_SPARSE_PRECOMPILE=0 \
        AIPERF_EXPERIMENTAL_FAST=0 \
        AIPERF_WARMUP_REQUESTS_PER_LANE=10 \
        DURATION=3600 \
        EVAL_ONLY="$eval_only" \
        RUN_EVAL="$eval_only" \
        EVAL_LIMIT="$eval_limit" \
        RUN_LABEL="$label" \
        sbatch \
            --parsable \
            --account="$slurm_account" \
            --partition="$slurm_partition" \
            --reservation="$slurm_reservation" \
            --exclusive \
            --gpus-per-node=8 \
            --nodelist="$node" \
            --job-name="${label:0:64}" \
            --time="$time_limit" \
            "${dependency_args[@]}" \
            --export=ALL \
            "$sbatch_script"
}

readonly initial_dependency="${AFTEROK_JOB_ID:-}"
smoke="$(submit_run "pr52664-${pr_sha:0:12}-smoke" "$initial_dependency" 04:00:00 1 true 8 none none)"
smoke="${smoke%%;*}"
gsm8k="$(submit_run "pr52664-${pr_sha:0:12}-gsm8k-full" "$smoke" 12:00:00 1 true full none none)"
gsm8k="${gsm8k%%;*}"
c1="$(submit_run "pr52664-${pr_sha:0:12}-c1" "$gsm8k" 06:00:00 1 false full none none)"
c1="${c1%%;*}"
c32="$(submit_run "pr52664-${pr_sha:0:12}-c32-offload" "$c1" 06:00:00 32 false full dram vllm-simple)"
c32="${c32%%;*}"

printf 'smoke=%s dependency=%s\n' "$smoke" "${initial_dependency:-none}"
printf 'gsm8k=%s dependency=%s\n' "$gsm8k" "$smoke"
printf 'c1=%s dependency=%s\n' "$c1" "$gsm8k"
printf 'c32_offload=%s dependency=%s\n' "$c32" "$c1"
