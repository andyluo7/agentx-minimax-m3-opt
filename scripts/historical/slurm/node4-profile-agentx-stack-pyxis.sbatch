#!/usr/bin/env bash
#SBATCH --job-name=minimaxm3-agentx-profile
#SBATCH --account=r7n
#SBATCH --partition=256C8G1H_MI355X_Ubuntu24
#SBATCH --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
#SBATCH --nodes=1
#SBATCH --nodelist=vultr-mi355x-6
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=256
#SBATCH --mem=3000000M
#SBATCH --time=02:00:00
#SBATCH --output=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.out
#SBATCH --error=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.err

set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly vllm_overlay="${VLLM_PROFILE_OVERLAY:-$shared_root/overlays/vllm-v0271-aiter-indexer-fused-ar-cache-router-e8e0d14}"
readonly vllm_eagle_overlay="$shared_root/overlays/vllm-agentx-draft-window"
readonly vllm_aiter_unified_attn_overlay="${VLLM_AITER_UNIFIED_ATTN_OVERLAY:-$shared_root/overlays/vllm-v0271-aiter-unified-attn-cg-metadata-fix-55f2b41}"
readonly vllm_warmup_overlay="${VLLM_MINIMAX_M3_WARMUP_OVERLAY:-$shared_root/overlays/vllm-v0271-minimaxm3-agentx-jit-warmup-v12-8d6f7a0}"
readonly aiter_overlay="$shared_root/overlays/aiter-sparse-indexer-cache-fusion-545d97c-266922c"
readonly mode="${PROFILE_MODE:-eagle3-l3}"
readonly result_dir="$shared_root/results/profile-$mode-${SLURM_JOB_ID}"
readonly runtime_root="/tmp/minimaxm3-agentx-profile-${SLURM_JOB_ID}"
readonly cache_root=/tmp/minimaxm3-agentx-profile-cache
readonly python_site=/usr/local/lib/python3.12/dist-packages
readonly vllm_warmup_site="$python_site/vllm/model_executor/warmup/minimax_m3_msa_warmup.py"
readonly flydsl_container_cache="$python_site/aiter/jit/flydsl_cache"

hostname
date -Is
uptime
free -h
df -h /tmp "$shared_root"
if [[ "${SLURM_GPUS_ON_NODE:-0}" != "8" ]]; then
    echo "Error: expected all 8 GPUs, got SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}" >&2
    exit 2
fi
ps -eo user,pid,ppid,%cpu,%mem,etime,args --sort=-%cpu | sed -n '1,40p'

mkdir -p \
    "$result_dir/torch-profiler" \
    "$runtime_root/tmp" \
    "$cache_root/home/.cache" \
    "$cache_root/home/.aiter/jit" \
    "$cache_root/flydsl-cache" \
    "$cache_root/triton"
cp "$shared_root/profile-agentx-stack.sh" "$runtime_root/profile-agentx-stack.sh"
chmod 0755 "$runtime_root/profile-agentx-stack.sh"

srun \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=8 \
    --container-image=vllm/vllm-openai-rocm:v0.27.1 \
    --container-mounts="$shared_root:$shared_root,$runtime_root:$runtime_root,$cache_root:$cache_root,$cache_root/flydsl-cache:$flydsl_container_cache,$vllm_overlay:$python_site/vllm/models/minimax_m3,$vllm_eagle_overlay/vllm/config/speculative.py:$python_site/vllm/config/speculative.py,$vllm_eagle_overlay/vllm/model_executor/models/llama_eagle3.py:$python_site/vllm/model_executor/models/llama_eagle3.py,$vllm_eagle_overlay/vllm/model_executor/models/llama.py:$python_site/vllm/model_executor/models/llama.py,$vllm_eagle_overlay/vllm/model_executor/layers/attention/attention.py:$python_site/vllm/model_executor/layers/attention/attention.py,$vllm_aiter_unified_attn_overlay/rocm_aiter_unified_attn.py:$python_site/vllm/v1/attention/backends/rocm_aiter_unified_attn.py,$vllm_warmup_overlay/minimax_m3_msa_warmup.py:$vllm_warmup_site,$aiter_overlay/aiter/ops/sparse_attention.py:$python_site/aiter/ops/sparse_attention.py,$aiter_overlay/aiter/ops/triton/attention/unified_attention.py:$python_site/aiter/ops/triton/attention/unified_attention.py,$aiter_overlay/aiter/ops/triton/_triton_kernels/attention/unified_attention.py:$python_site/aiter/ops/triton/_triton_kernels/attention/unified_attention.py,$aiter_overlay/aiter/ops/minimax_m3_fused_qknorm_rope.py:$python_site/aiter/ops/minimax_m3_fused_qknorm_rope.py,$aiter_overlay/aiter/jit/optCompilerConfig.json:$python_site/aiter/jit/optCompilerConfig.json,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    env \
        HOME="$cache_root/home" \
        XDG_CACHE_HOME="$cache_root/home/.cache" \
        HF_HOME="$shared_root/storage/hf" \
        HF_HUB_CACHE="$shared_root/storage/hf/hub" \
        TRANSFORMERS_CACHE="$shared_root/storage/hf/hub" \
        TMPDIR="$runtime_root/tmp" \
        TRITON_CACHE_DIR="$cache_root/triton" \
        PYTHONPATH="$aiter_overlay" \
        AITER_JIT_DIR="$cache_root/home/.aiter/jit" \
        FLYDSL_RUNTIME_CACHE_DIR="$flydsl_container_cache" \
        AITER_FUSED_CACHE_INSERT_OVERLAY="$aiter_overlay" \
        VLLM_MINIMAX_M3_FUSED_CACHE_INSERT="${VLLM_MINIMAX_M3_FUSED_CACHE_INSERT:-1}" \
        VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA="${VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA:-1}" \
        VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP="${VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP:-1}" \
        VLLM_AITER_UNIFIED_ATTN_OVERLAY="$vllm_aiter_unified_attn_overlay" \
        VLLM_MINIMAX_M3_WARMUP_OVERLAY="$vllm_warmup_overlay" \
        VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY="${VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY:-1}" \
        VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT="${VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT:-1}" \
        VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL="${VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL:-aiter}" \
        VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER="${VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER:-aiter}" \
        VLLM_MINIMAX_M3_ASM_SPARSE_ATTN="${VLLM_MINIMAX_M3_ASM_SPARSE_ATTN:-0}" \
        VLLM_USE_BREAKABLE_CUDAGRAPH="${BREAKABLE_CUDAGRAPH:-0}" \
        VLLM_ROCM_USE_AITER=1 \
        VLLM_ROCM_USE_AITER_MOE=1 \
        VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
        VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 \
        VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
        VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
        VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB=256 \
        VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB="${VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB:-2048}" \
        VLLM_ENGINE_READY_TIMEOUT_S=3600 \
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
        VLLM_SERVER_DEV_MODE=1 \
        PROFILE_MODE="$mode" \
        TARGET_ATTENTION_BACKEND="${TARGET_ATTENTION_BACKEND:-ROCM_AITER_UNIFIED_ATTN}" \
        DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-ROCM_AITER_UNIFIED_ATTN}" \
        DRAFT_ATTENTION_WINDOW="${DRAFT_ATTENTION_WINDOW:-32768}" \
        AITER_UNIFIED_ATTN_SLIDING_DECODE_3D="${AITER_UNIFIED_ATTN_SLIDING_DECODE_3D:-1}" \
        PROFILE_SYNTHETIC_ACCEPTANCE="${PROFILE_SYNTHETIC_ACCEPTANCE:-0}" \
        RESULT_DIR="$result_dir" \
        PROFILE_PROMPT_TOKENS="${PROFILE_PROMPT_TOKENS:-262144}" \
        PROFILE_WARM_PROMPT_TOKENS="${PROFILE_WARM_PROMPT_TOKENS:-${PROFILE_PROMPT_TOKENS:-262144}}" \
        PROFILE_MAX_TOKENS="${PROFILE_MAX_TOKENS:-128}" \
        PORT="${PORT:-8886}" \
        bash "$runtime_root/profile-agentx-stack.sh"

test -s "$result_dir/profile-request-summary.json"
test -s "$result_dir/profile-metadata.txt"
find "$result_dir/torch-profiler" -type f -size +0c -print | sort
date -Is
