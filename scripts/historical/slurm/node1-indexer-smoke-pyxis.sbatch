#!/usr/bin/env bash
#SBATCH --job-name=minimaxm3-indexer-smoke-n1
#SBATCH --account=r7n
#SBATCH --partition=256C8G1H_MI355X_Ubuntu24
#SBATCH --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
#SBATCH --nodes=1
#SBATCH --nodelist=vultr-mi355x-1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.out
#SBATCH --error=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.err

set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly overlay_root="$shared_root/overlays/aiter-sparse-indexer-545d97c"
readonly runtime_root=/tmp/minimaxm3-indexer-smoke
readonly sparse_wrapper_site=/usr/local/lib/python3.12/dist-packages/aiter/ops/sparse_attention.py

hostname
date -Is
who || true
uptime
free -h
df -h /tmp "$shared_root"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showuse --showmemuse --showpids
elif [[ -x /opt/rocm/bin/rocm-smi ]]; then
    /opt/rocm/bin/rocm-smi --showuse --showmemuse --showpids
else
    echo "host rocm-smi unavailable; container smoke will validate GPU access"
fi
ps -eo user,pid,ppid,%cpu,%mem,etime,args --sort=-%cpu | head -40

mkdir -p "$runtime_root/home/.cache" "$runtime_root/tmp" "$runtime_root/triton"

srun \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --container-image=vllm/vllm-openai-rocm:v0.27.1 \
    --container-mounts="$shared_root:$shared_root,$runtime_root:$runtime_root,$overlay_root/aiter/ops/sparse_attention.py:$sparse_wrapper_site,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    --container-env=HOME,XDG_CACHE_HOME,TRITON_CACHE_DIR,TMPDIR,PYTHONPATH,AITER_ROOT_DIR,AITER_JIT_DIR,GPU_ARCHS \
    --export=ALL,HOME="$runtime_root/home",XDG_CACHE_HOME="$runtime_root/home/.cache",TRITON_CACHE_DIR="$runtime_root/triton",TMPDIR="$runtime_root/tmp",PYTHONPATH="$overlay_root",AITER_ROOT_DIR="$runtime_root/home/.aiter",AITER_JIT_DIR="$runtime_root/home/.aiter/jit",GPU_ARCHS=gfx950 \
    env \
        HOME="$runtime_root/home" \
        XDG_CACHE_HOME="$runtime_root/home/.cache" \
        TRITON_CACHE_DIR="$runtime_root/triton" \
        TMPDIR="$runtime_root/tmp" \
        PYTHONPATH="$overlay_root" \
        AITER_ROOT_DIR="$runtime_root/home/.aiter" \
        AITER_JIT_DIR="$runtime_root/home/.aiter/jit" \
        INDEXER_SMOKE_DTYPE="${INDEXER_SMOKE_DTYPE:-both}" \
        GPU_ARCHS=gfx950 \
        python3 "$shared_root/indexer_smoke.py"
