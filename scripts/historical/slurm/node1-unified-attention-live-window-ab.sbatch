#!/usr/bin/env bash
#SBATCH --job-name=aiter-ua-live-window-ab
#SBATCH --account=r7n
#SBATCH --partition=256C8G1H_MI355X_Ubuntu24
#SBATCH --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
#SBATCH --nodes=1
#SBATCH --nodelist=vultr-mi355x-1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.out
#SBATCH --error=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/logs/slurm-%j.err

set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly source_root="$shared_root/upstream-ab/aiter-35c652ed3"
readonly python_site=/usr/local/lib/python3.12/dist-packages
readonly runtime_root=/tmp/aiter-ua-live-window-ab-${SLURM_JOB_ID}

hostname
date -Is
who || true
uptime
free -h
df -h /tmp "$shared_root"
if [[ "${SLURM_GPUS_ON_NODE:-0}" != "8" ]]; then
    echo "Error: expected all 8 GPUs, got SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}" >&2
    exit 2
fi
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showuse --showmemuse --showpids
elif [[ -x /opt/rocm/bin/rocm-smi ]]; then
    /opt/rocm/bin/rocm-smi --showuse --showmemuse --showpids
else
    echo "host rocm-smi unavailable; the container run will validate GPU access"
fi
ps -eo user,pid,ppid,%cpu,%mem,etime,args --sort=-%cpu | sed -n '1,40p'

for used_file in /sys/class/drm/card*/device/mem_info_vram_used; do
    [[ -r "$used_file" ]] || continue
    used_bytes=$(<"$used_file")
    if (( used_bytes > 1073741824 )); then
        echo "Error: pre-existing GPU VRAM use is above 1 GiB: $used_file=$used_bytes" >&2
        exit 3
    fi
done
if command -v fuser >/dev/null 2>&1; then
    kfd_users=$(fuser /dev/kfd 2>/dev/null || true)
    if [[ -n "$kfd_users" ]]; then
        echo "Error: pre-existing /dev/kfd users: $kfd_users" >&2
        exit 4
    fi
fi

run_variant() {
    local label="$1"
    local variant_root="$source_root/$label"
    local variant_runtime="$runtime_root/$label"
    mkdir -p \
        "$variant_runtime/home/.cache" \
        "$variant_runtime/tmp" \
        "$variant_runtime/triton"
    srun \
        --nodes=1 \
        --ntasks=1 \
        --gpus-per-task=1 \
        --container-image=vllm/vllm-openai-rocm:v0.27.1 \
        --container-mounts="$shared_root:$shared_root,$variant_runtime:$variant_runtime,$variant_root/aiter/ops/triton/attention/unified_attention.py:$python_site/aiter/ops/triton/attention/unified_attention.py,$variant_root/aiter/ops/triton/_triton_kernels/attention/unified_attention.py:$python_site/aiter/ops/triton/_triton_kernels/attention/unified_attention.py,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
        env \
            HOME="$variant_runtime/home" \
            XDG_CACHE_HOME="$variant_runtime/home/.cache" \
            TRITON_CACHE_DIR="$variant_runtime/triton" \
            TMPDIR="$variant_runtime/tmp" \
            GPU_ARCHS=gfx950 \
            python3 "$shared_root/unified_attention_live_window_ab.py" \
                --label "$label" \
                --batch-sizes "${AITER_UA_BATCH_SIZES:-1:8:32}" \
                --sequence-lengths "${AITER_UA_SEQUENCE_LENGTHS:-8192:32768:81920:262144}"
}

run_variant current-main
run_variant live-window
