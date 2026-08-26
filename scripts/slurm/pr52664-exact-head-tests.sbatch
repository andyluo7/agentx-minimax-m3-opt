#!/usr/bin/env bash
#SBATCH --job-name=pr52664-exact-head-tests
#SBATCH --account=r7n
#SBATCH --partition=256C8G1H_MI355X_Ubuntu24
#SBATCH --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/upstream-validation/logs/pr52664-tests-%j.out
#SBATCH --error=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/upstream-validation/logs/pr52664-tests-%j.err

set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly validation_root="$shared_root/upstream-validation"
readonly vllm_sha=16c5a20643e93f0ab5cf364c59bfcb0a90d72b2c
readonly aiter_sha=cb3c7a628645dd9b03610d36b543d39225e2cde5
readonly vllm_root="$validation_root/vllm-${vllm_sha:0:12}"
readonly aiter_root="$validation_root/aiter-${aiter_sha:0:12}"
readonly runtime_root="/tmp/pr52664-exact-head-tests-${SLURM_JOB_ID}"
readonly image=vllm/vllm-openai-rocm:v0.27.1

mkdir -p \
    "$validation_root/logs" \
    "$runtime_root/home/.cache" \
    "$runtime_root/home/.aiter/jit" \
    "$runtime_root/tmp" \
    "$runtime_root/triton"

test "$(git -C "$vllm_root" rev-parse HEAD)" = "$vllm_sha"
test "$(git -C "$aiter_root" rev-parse HEAD)" = "$aiter_sha"
test "${SLURM_GPUS_ON_NODE:-0}" = 8

for used_file in /sys/class/drm/card*/device/mem_info_vram_used; do
    [[ -r "$used_file" ]] || continue
    used_bytes=$(<"$used_file")
    if (( used_bytes > 1073741824 )); then
        echo "Error: pre-existing GPU VRAM use exceeds 1 GiB: $used_file=$used_bytes" >&2
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

date -Is
hostname
printf 'vllm_sha=%s\naiter_sha=%s\nimage=%s\n' \
    "$vllm_sha" "$aiter_sha" "$image"
if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showuniqueid --showmeminfo vram
elif [[ -x /opt/rocm/bin/rocm-smi ]]; then
    /opt/rocm/bin/rocm-smi --showproductname --showuniqueid --showmeminfo vram
else
    echo "host rocm-smi unavailable; recording DRM product and VRAM state"
    for product_file in /sys/class/drm/card*/device/product_name; do
        [[ -r "$product_file" ]] || continue
        printf '%s ' "$product_file"
        cat "$product_file"
    done
    for used_file in /sys/class/drm/card*/device/mem_info_vram_used; do
        [[ -r "$used_file" ]] || continue
        printf '%s ' "$used_file"
        cat "$used_file"
    done
fi

# The single-quoted command is expanded by the container shell from env above.
# shellcheck disable=SC2016
srun \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=8 \
    --container-image="$image" \
    --container-mounts="$shared_root:$shared_root,$runtime_root:$runtime_root,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri" \
    env \
        HOME="$runtime_root/home" \
        XDG_CACHE_HOME="$runtime_root/home/.cache" \
        AITER_ROOT_DIR="$runtime_root/home/.aiter" \
        AITER_JIT_DIR="$runtime_root/home/.aiter/jit" \
        AITER_META_DIR="$aiter_root" \
        TRITON_CACHE_DIR="$runtime_root/triton" \
        TMPDIR="$runtime_root/tmp" \
        PYTHONPYCACHEPREFIX="$runtime_root/tmp/pycache" \
        PYTHONPATH="$vllm_root:$aiter_root" \
        VLLM_TEST_VENV="$runtime_root/.venv" \
        VLLM_SOURCE_ROOT="$vllm_root" \
        AITER_SOURCE_ROOT="$aiter_root" \
        GPU_ARCHS=gfx950 \
        bash -lc '
            set -euo pipefail
            uv venv --system-site-packages "$VLLM_TEST_VENV"
            "$VLLM_TEST_VENV/bin/python" - <<"PY"
import aiter
import vllm

print(f"vllm_module={vllm.__file__}")
print(f"aiter_module={aiter.__file__}")
PY
            "$VLLM_TEST_VENV/bin/python" -m pytest -q \
                "$VLLM_SOURCE_ROOT/tests/kernels/attention/test_minimax_m3_indexer_selection.py"
            "$VLLM_TEST_VENV/bin/python" -m pytest -q \
                "$VLLM_SOURCE_ROOT/tests/kernels/attention/test_minimax_m3.py" \
                -k "aiter_indexer_matches_reference_and_page_table_builders or aiter_sparse_block_table_handles_padded_decode_rows or aiter_decode_sparse_block_table_supports_spec_decode"
        '

date -Is
