#!/usr/bin/env bash
set -euo pipefail

readonly shared_root="${AGENTX_SHARED_ROOT:-/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx}"
readonly sbatch_script="${AGENTX_SBATCH_SCRIPT:-$shared_root/run-agentx.sbatch}"
readonly recipe="${AGENTX_RECIPE_PATH:-$shared_root/repos/InferenceX/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh}"
readonly sweep_id="${SWEEP_ID_OVERRIDE:-full-indexer-complete-curve-20260825-r1}"
readonly slurm_account="${SLURM_ACCOUNT:-r7n}"
readonly slurm_partition="${SLURM_PARTITION:-256C8G1H_MI355X_Ubuntu24}"
readonly slurm_reservation="${SLURM_RESERVATION:-aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation}"
readonly node_a="${MI355X_NODE_A:-vultr-mi355x-4}"
readonly node_b="${MI355X_NODE_B:-vultr-mi355x-5}"
readonly node_c="${MI355X_NODE_C:-vultr-mi355x-6}"

submit_point() {
    local node="$1"
    local conc="$2"
    local kv_offloading="$3"
    local kv_backend="$4"
    local dependency="${5:-}"
    local policy label container_name output

    if [[ "$node" == "vultr-mi355x-6" ]]; then
        container_name=minimaxm3-vllm-rocm-v0271
    else
        container_name=pyxis_minimaxm3-vllm-rocm-v0271
    fi

    if [[ "$kv_offloading" == "none" ]]; then
        policy=resident
    else
        policy=vllm-simple
    fi
    label="mtp-k4-aiter-full-indexer-${policy}-v19-tp4-c${conc}-${sweep_id}"

    local dependency_args=()
    if [[ -n "$dependency" ]]; then
        dependency_args+=(--dependency="afterok:$dependency")
    fi

    output="$(
        env \
            -u SYNTHETIC_ACCEPT_LEN \
            -u ALLOW_NON_GOLDEN_ACCEPTANCE \
            -u DRAFT_MAX_MODEL_LEN \
            -u DISABLE_SPECULATIVE_DECODING \
            -u AITER_CONFIG_GEMM_BF16 \
            -u AITER_CONFIG_FMOE \
            -u VLLM_MINIMAX_M3_OVERLAY \
            -u VLLM_EAGLE3_OVERLAY \
            -u VLLM_AITER_UNIFIED_ATTN_OVERLAY \
            -u AITER_INDEXER_OVERLAY \
            -u VLLM_MINIMAX_M3_WARMUP_OVERLAY \
            -u AITER_FUSED_CACHE_INSERT_OVERLAY \
            -u EVAL_LIMIT \
            AGENTX_CONTAINER_NAME="$container_name" \
            AGENTX_SHARED_ROOT="$shared_root" \
            AGENTX_RECIPE_OVERRIDE="$recipe" \
            TP=4 \
            MAX_NUM_SEQS=256 \
            MAX_NUM_BATCHED_TOKENS=32768 \
            MAX_CUDAGRAPH_CAPTURE_SIZE=512 \
            GPU_MEMORY_UTILIZATION=0.85 \
            AIPERF_EXPERIMENTAL_FAST=0 \
            AIPERF_WARMUP_REQUESTS_PER_LANE=10 \
            KV_OFFLOADING="$kv_offloading" \
            KV_OFFLOAD_BACKEND="$kv_backend" \
            TOTAL_CPU_DRAM_GB=1499 \
            TARGET_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \
            DRAFT_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \
            DRAFT_ATTENTION_WINDOW=32768 \
            AITER_UNIFIED_ATTN_SLIDING_DECODE_3D=1 \
            INDEXER_KV_DTYPE=fp8 \
            KV_CACHE_DTYPE=fp8 \
            VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY=1 \
            VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT=1 \
            VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL=aiter \
            VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER=aiter \
            VLLM_MINIMAX_M3_FUSED_CACHE_INSERT=1 \
            VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA=1 \
            VLLM_MINIMAX_M3_ROCM_FP32_ROUTER_GEMM=0 \
            VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP=1 \
            JIT_MONITOR_VERBOSE=1 \
            VLLM_MINIMAX_M3_ASM_SPARSE_ATTN=0 \
            AITER_SPARSE_PRECOMPILE=0 \
            VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048 \
            VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 \
            ENABLE_AGENTX_POWER=0 \
            NUM_SPEC_TOKENS=4 \
            CONC="$conc" \
            DURATION=3600 \
            EVAL_ONLY=false \
            RUN_EVAL=false \
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
                --time=06:00:00 \
                "${dependency_args[@]}" \
                --export=ALL \
                "$sbatch_script"
    )"
    printf '%s\n' "${output%%;*}"
}

# Three exclusive-node chains. Offload points run first; resident points fill
# the final wave. Dependencies avoid CPU-memory and GPU contention per node.
c15="$(submit_point "$node_a" 15 dram vllm-simple)"
c20="$(submit_point "$node_b" 20 dram vllm-simple)"
c25="$(submit_point "$node_c" 25 dram vllm-simple)"

c30="$(submit_point "$node_a" 30 dram vllm-simple "$c15")"
c32="$(submit_point "$node_b" 32 dram vllm-simple "$c20")"
c10="$(submit_point "$node_c" 10 none none "$c25")"

c1="$(submit_point "$node_a" 1 none none "$c30")"
c5="$(submit_point "$node_b" 5 none none "$c32")"

printf 'sweep_id=%s\n' "$sweep_id"
printf 'c15_offload_tp4=%s\n' "$c15"
printf 'c20_offload_tp4=%s\n' "$c20"
printf 'c25_offload_tp4=%s\n' "$c25"
printf 'c30_offload_tp4=%s dependency=%s\n' "$c30" "$c15"
printf 'c32_offload_tp4=%s dependency=%s\n' "$c32" "$c20"
printf 'c10_resident_tp4=%s dependency=%s\n' "$c10" "$c25"
printf 'c1_resident_tp4=%s dependency=%s\n' "$c1" "$c30"
printf 'c5_resident_tp4=%s dependency=%s\n' "$c5" "$c32"
