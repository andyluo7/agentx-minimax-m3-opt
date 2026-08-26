#!/usr/bin/env bash
set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly sbatch_script="$shared_root/node6-agentx.sbatch"
readonly recipe="$shared_root/repos/InferenceX/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp_full_indexer.sh"
readonly warmup_overlay="$shared_root/overlays/vllm-v0271-minimaxm3-agentx-jit-warmup-v23-tp4-prod-buffer"
readonly dependency_job="${DEPENDENCY_JOB:-1411}"
readonly label=mtp-k4-aiter-full-indexer-vllm-simple-v23-tp4-c15-full-indexer-complete-curve-20260825-r2

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
        KV_OFFLOADING=dram \
        KV_OFFLOAD_BACKEND=vllm-simple \
        TOTAL_CPU_DRAM_GB=1499 \
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
        VLLM_MINIMAX_M3_WARMUP_OVERLAY="$warmup_overlay" \
        JIT_MONITOR_VERBOSE=1 \
        VLLM_MINIMAX_M3_ASM_SPARSE_ATTN=0 \
        AITER_SPARSE_PRECOMPILE=1 \
        VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB=2048 \
        VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 \
        ENABLE_AGENTX_POWER=0 \
        NUM_SPEC_TOKENS=4 \
        CONC=15 \
        DURATION=3600 \
        EVAL_ONLY=false \
        RUN_EVAL=false \
        RUN_LABEL="$label" \
        sbatch \
            --parsable \
            --exclusive \
            --nodelist=vultr-mi355x-6 \
            --job-name="${label:0:64}" \
            --time=06:00:00 \
            --dependency="afterany:$dependency_job" \
            --export=ALL \
            "$sbatch_script"
)"

printf 'c15_full_indexer_v23_rerun=%s dependency=%s\n' \
    "${output%%;*}" "$dependency_job"
