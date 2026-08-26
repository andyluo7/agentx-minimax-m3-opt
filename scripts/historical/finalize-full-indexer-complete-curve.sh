#!/usr/bin/env bash
set -euo pipefail

readonly project_root=/Users/anluo/code/minimaxm3-agentx-70pct
readonly artifact_root="$project_root/artifacts/full-indexer-complete-curve-20260825-r1"
readonly remote=andy_luo_3v7@64.177.9.51
readonly remote_results=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx/results
readonly output_prefix=/Users/anluo/Documents/minimax-m3-agentx-pareto-full-indexer-20260825-r1
readonly validation_report="$artifact_root/validation-report.json"

runs=(
    mtp-k4-aiter-full-indexer-resident-v19-tp4-c1-full-indexer-complete-curve-20260825-r1
    mtp-k4-aiter-full-indexer-resident-v19-tp4-c5-full-indexer-complete-curve-20260825-r1
    mtp-k4-aiter-full-indexer-resident-v19-tp4-c10-full-indexer-complete-curve-20260825-r1
    mtp-k4-aiter-full-indexer-vllm-simple-v19-tp4-c15-full-indexer-complete-curve-20260825-r1
    mtp-k4-aiter-full-indexer-vllm-simple-v23-tp4-c15-full-indexer-complete-curve-20260825-r2
    mtp-k4-aiter-full-indexer-vllm-simple-v19-tp4-c20-full-indexer-complete-curve-20260825-r1
    mtp-k4-aiter-full-indexer-vllm-simple-v19-tp4-c25-full-indexer-complete-curve-20260825-r1
    mtp-k4-aiter-full-indexer-vllm-simple-v19-tp4-c30-full-indexer-complete-curve-20260825-r1
    mtp-k4-aiter-full-indexer-vllm-simple-v19-tp4-c32-full-indexer-complete-curve-20260825-r1
)

for suffix in png pdf csv; do
    if [[ -e "$output_prefix.$suffix" ]]; then
        printf 'Refusing to overwrite existing output: %s.%s\n' "$output_prefix" "$suffix" >&2
        exit 2
    fi
done

mkdir -p "$artifact_root"
for run in "${runs[@]}"; do
    mkdir -p "$artifact_root/$run"
    rsync -a --prune-empty-dirs \
        --include='*/' \
        --include='run-metadata.txt' \
        --include='benchmark_command.txt' \
        --include='vllm_command.txt' \
        --include='console.log' \
        --include='server.log' \
        --include='profile_export.jsonl' \
        --include='aiperf.log' \
        --include="${run}.json" \
        --exclude='*' \
        "$remote:$remote_results/$run/" \
        "$artifact_root/$run/"
done

python3 "$project_root/validate_full_indexer_complete_curve.py" \
    "$artifact_root" \
    --output "$validation_report"

MPLCONFIGDIR=/tmp/mpl-minimax-full-indexer-curve \
    python3 "$project_root/plot_agentx_pareto.py" \
        --artifact-root "$artifact_root" \
        --output-prefix "$output_prefix"

printf 'validation=%s\n' "$validation_report"
printf 'png=%s.png\n' "$output_prefix"
printf 'pdf=%s.pdf\n' "$output_prefix"
printf 'csv=%s.csv\n' "$output_prefix"
