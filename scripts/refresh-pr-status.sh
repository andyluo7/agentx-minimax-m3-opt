#!/usr/bin/env bash
set -euo pipefail

command -v gh >/dev/null 2>&1 || {
    echo "gh is required" >&2
    exit 2
}

prs=(
    "vllm-project/vllm 53821 focused"
    "vllm-project/vllm 53827 focused"
    "ROCm/aiter 5004 focused"
    "vllm-project/vllm 53833 focused"
    "vllm-project/vllm 53695 dependency"
    "ROCm/aiter 4813 dependency"
    "vllm-project/vllm 52849 dependency"
    "vllm-project/vllm 52664 dependency"
    "ROCm/aiter 4787 dependency"
    "vllm-project/vllm 49170 dependency"
    "vllm-project/vllm 47270 dependency"
    "ROCm/aiter 4918 dependency"
    "vllm-project/vllm 53448 alternative"
    "vllm-project/vllm 52668 alternative"
    "SemiAnalysisAI/InferenceX 2726 superseded"
)

printf 'group\trepository\tpr\tstate\tdraft\tmerge_state\treview\thead\tupdated\turl\ttitle\n'
for entry in "${prs[@]}"; do
    read -r repo number group <<<"$entry"
    gh pr view "$number" --repo "$repo" \
        --json number,state,isDraft,mergeStateStatus,reviewDecision,headRefOid,updatedAt,url,title \
        --jq '["'"$group"'", "'"$repo"'", (.number|tostring), .state, (.isDraft|tostring), .mergeStateStatus, (.reviewDecision // ""), .headRefOid[0:12], .updatedAt, .url, .title] | @tsv'
done
