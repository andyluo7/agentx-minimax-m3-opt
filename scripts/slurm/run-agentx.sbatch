#!/usr/bin/env bash
#SBATCH --job-name=minimaxm3-agentx
#SBATCH --account=r7n
#SBATCH --partition=256C8G1H_MI355X_Ubuntu24
#SBATCH --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=256
#SBATCH --mem=3000000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

readonly shared_root="${AGENTX_SHARED_ROOT:-/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx}"

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
    echo "host rocm-smi unavailable; the container startup log must validate all GPUs"
fi
ps -eo user,pid,ppid,%cpu,%mem,etime,args --sort=-%cpu | sed -n '1,40p'
enroot list || true

export CONC="${CONC:-1}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$CONC}"
export RUN_LABEL="${RUN_LABEL:-resident-tp4-c${CONC}-maxseq${MAX_NUM_SEQS}-fast}"
export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-1}"

readonly local_runner="/tmp/minimaxm3-agentx-runner-${SLURM_JOB_ID}.sh"
install -m 0700 \
    "$shared_root/run-agentx.sh" \
    "$local_runner"
exec "$local_runner"
