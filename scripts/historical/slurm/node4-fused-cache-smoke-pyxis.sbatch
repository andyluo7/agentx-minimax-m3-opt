#!/usr/bin/env bash
#SBATCH --job-name=minimaxm3-fused-cache-smoke
#SBATCH --account=r7n
#SBATCH --partition=256C8G1H_MI355X_Ubuntu24
#SBATCH --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
#SBATCH --nodes=1
#SBATCH --nodelist=vultr-mi355x-4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.out
#SBATCH --error=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.err

set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly aiter_overlay="$shared_root/overlays/aiter-sparse-indexer-cache-fusion-545d97c-266922c"
readonly vllm_overlay="$shared_root/overlays/vllm-v0271-aiter-indexer-cache-fusion-e41bdb0"
readonly runtime_root=/tmp/minimaxm3-fused-cache-smoke
readonly python_site=/usr/local/lib/python3.12/dist-packages

hostname
date -Is
free -h
df -h /tmp "$shared_root"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showuse --showmemuse --showpids
elif [[ -x /opt/rocm/bin/rocm-smi ]]; then
    /opt/rocm/bin/rocm-smi --showuse --showmemuse --showpids
else
    echo "host rocm-smi unavailable; container smoke will validate GPU access"
fi

mkdir -p \
    "$runtime_root/home/.cache" \
    "$runtime_root/tmp" \
    "$runtime_root/triton"

srun \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --container-image=vllm/vllm-openai-rocm:v0.27.1 \
    --container-mounts="$shared_root:$shared_root,$runtime_root:$runtime_root,$vllm_overlay:$python_site/vllm/models/minimax_m3,$aiter_overlay/aiter/ops/sparse_attention.py:$python_site/aiter/ops/sparse_attention.py,$aiter_overlay/aiter/ops/minimax_m3_fused_qknorm_rope.py:$python_site/aiter/ops/minimax_m3_fused_qknorm_rope.py,$aiter_overlay/aiter/jit/optCompilerConfig.json:$python_site/aiter/jit/optCompilerConfig.json,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    env \
        HOME="$runtime_root/home" \
        XDG_CACHE_HOME="$runtime_root/home/.cache" \
        TRITON_CACHE_DIR="$runtime_root/triton" \
        TMPDIR="$runtime_root/tmp" \
        PYTHONPATH="$aiter_overlay" \
        AITER_ROOT_DIR="$runtime_root/home/.aiter" \
        AITER_JIT_DIR="$runtime_root/home/.aiter/jit" \
        AITER_FUSED_CACHE_INSERT_OVERLAY="$aiter_overlay" \
        GPU_ARCHS=gfx950 \
        HIP_VISIBLE_DEVICES=0 \
        CUDA_VISIBLE_DEVICES=0 \
        ROCR_VISIBLE_DEVICES=0 \
        python3 "$shared_root/fused_cache_insert_smoke.py"
