#!/usr/bin/env bash
set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly sbatch_script="$shared_root/node6-agentx.sbatch"
readonly recipe="$shared_root/repos/InferenceX/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp_full_indexer.sh"
readonly sweep_id="${SWEEP_ID_OVERRIDE:-full-indexer-a2a-20260825-r2}"
readonly node=vultr-mi355x-6

submit_point() {
    local conc="$1"
    local kv_offloading="$2"
    local kv_backend="$3"
    local dependency="${4:-}"
    local policy label output

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
            -u EVAL_LIMIT \
            AGENTX_CONTAINER_NAME=minimaxm3-vllm-rocm-v0271 \
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
            TOTAL_CPU_DRAM_GB=1547 \
            TARGET_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \
            DRAFT_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \
            DRAFT_ATTENTION_WINDOW=32768 \
            AITER_UNIFIED_ATTN_SLIDING_DECODE_3D=1 \
            INDEXER_KV_DTYPE=fp8 \
            KV_CACHE_DTYPE=fp8 \
            VLLM_MINIMAX_M3_OVERLAY="$shared_root/overlays/vllm-v0271-aiter-indexer-fused-ar-cache-router-e8e0d14" \
            VLLM_EAGLE3_OVERLAY="$shared_root/overlays/vllm-agentx-draft-window" \
            VLLM_AITER_UNIFIED_ATTN_OVERLAY="$shared_root/overlays/vllm-v0271-aiter-unified-attn-cg-metadata-fix-55f2b41" \
            VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY=1 \
            VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT=1 \
            VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL=aiter \
            VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER=aiter \
            AITER_INDEXER_OVERLAY="$shared_root/overlays/aiter-sparse-indexer-cache-fusion-545d97c-266922c" \
            VLLM_MINIMAX_M3_FUSED_CACHE_INSERT=1 \
            VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA=1 \
            VLLM_MINIMAX_M3_ROCM_FP32_ROUTER_GEMM=0 \
            VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP=1 \
            VLLM_MINIMAX_M3_WARMUP_OVERLAY="$shared_root/overlays/vllm-v0271-minimaxm3-agentx-jit-warmup-v19-cc39c7f" \
            JIT_MONITOR_VERBOSE=1 \
            VLLM_MINIMAX_M3_ASM_SPARSE_ATTN=0 \
            AITER_SPARSE_PRECOMPILE=1 \
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
                --exclusive \
                --nodelist="$node" \
                --job-name="${label:0:64}" \
                --time=06:00:00 \
                "${dependency_args[@]}" \
                --export=ALL \
                "$sbatch_script"
    )"
    printf '%s\n' "${output%%;*}"
}

case "${FULL_INDEXER_POINT:-both}" in
    c1)
        c1="$(submit_point 1 none none)"
        printf 'sweep_id=%s\n' "$sweep_id"
        printf 'c1_resident_tp4=%s\n' "$c1"
        ;;
    c32)
        : "${C1_JOB_ID:?Set C1_JOB_ID to the successful C1 Slurm job ID.}"
        c32="$(submit_point 32 dram vllm-simple "$C1_JOB_ID")"
        printf 'sweep_id=%s\n' "$sweep_id"
        printf 'c32_simple_tp4=%s dependency=%s\n' "$c32" "$C1_JOB_ID"
        ;;
    both)
        c1="$(submit_point 1 none none)"
        c32="$(submit_point 32 dram vllm-simple "$c1")"
        printf 'sweep_id=%s\n' "$sweep_id"
        printf 'c1_resident_tp4=%s\n' "$c1"
        printf 'c32_simple_tp4=%s dependency=%s\n' "$c32" "$c1"
        ;;
    *)
        printf 'Unsupported FULL_INDEXER_POINT=%s (expected c1, c32, or both).\n' \
            "$FULL_INDEXER_POINT" >&2
        exit 2
        ;;
esac
