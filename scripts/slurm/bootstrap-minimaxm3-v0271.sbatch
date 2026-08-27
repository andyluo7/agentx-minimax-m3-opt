#!/usr/bin/env bash
#SBATCH --job-name=bootstrap-minimaxm3-v0271
#SBATCH --account=r7n
#SBATCH --partition=256C8G1H_MI355X_Ubuntu24
#SBATCH --reservation=aac17_vultr-mi355x-1_vultr-mi355x-2_vultr-mi355x-3_vultr-mi355x-4_vultr-mi355x-5_vultr-mi355x-6_reservation
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/upstream-validation/logs/bootstrap-minimaxm3-v0271-%j.out
#SBATCH --error=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/upstream-validation/logs/bootstrap-minimaxm3-v0271-%j.err

set -euo pipefail

readonly image_digest=sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7
readonly image_tag=vllm/vllm-openai-rocm:v0.27.1
readonly image_ref="docker://$image_tag"
readonly image_name=minimaxm3-vllm-rocm-v0271-bb44b39a
readonly image_dir=/data/enroot/andy_luo_3v7
readonly image_file="$image_dir/$image_name.sqsh"
readonly image_tmp="$image_file.partial.${SLURM_JOB_ID:-$$}"
readonly enroot_rc=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/enroot-command.rc

mkdir -p "$image_dir"
printf 'hostname=%s\nimage_tag=%s\nimage_digest_expected=%s\nimage_name=%s\n' \
    "$(hostname)" "$image_tag" "$image_digest" "$image_name"

registry_token="$({
    curl -fsSL \
        'https://auth.docker.io/token?service=registry.docker.io&scope=repository:vllm/vllm-openai-rocm:pull'
} | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')"
registry_digest="$({
    curl -fsSI \
        -H "Authorization: Bearer $registry_token" \
        -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json' \
        'https://registry-1.docker.io/v2/vllm/vllm-openai-rocm/manifests/v0.27.1'
} | awk -F': ' 'tolower($1) == "docker-content-digest" {gsub("\\r", "", $2); print $2}')"
printf 'image_digest_resolved=%s\n' "$registry_digest"
if [[ "$registry_digest" != "$image_digest" ]]; then
    echo "ERROR: image tag resolved to an unexpected registry digest" >&2
    exit 2
fi

if ! unsquashfs -s "$image_file" >/dev/null 2>&1; then
    trap 'rm -f "$image_tmp"' EXIT
    enroot import -o "$image_tmp" "$image_ref"
    unsquashfs -s "$image_tmp"
    mv "$image_tmp" "$image_file"
    trap - EXIT
fi

if ! enroot list | grep -Fx "$image_name" >/dev/null; then
    enroot create --name "$image_name" "$image_file"
fi

enroot list | grep -Fx "$image_name"
enroot start --rc "$enroot_rc" "$image_name" \
    bash -lc 'python -c "import vllm; print(vllm.__version__)"'
