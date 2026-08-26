#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
    echo "Usage: $0 CANONICAL_ARTIFACT_ROOT GSM8K_ARTIFACT_DIR [INFERENCEX_ROOT]" >&2
    exit 2
fi

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly canonical_root="$1"
readonly eval_dir="$2"
readonly inferencex_root="${3:-/Users/anluo/code/InferenceX-minimaxm3-mi355x-gap}"
readonly performance_report="$(mktemp /tmp/minimaxm3-agentx-performance.XXXXXX.json)"
trap 'rm -f "$performance_report"' EXIT

python3 "$script_dir/verify_70pct.py" "$canonical_root" >"$performance_report"
jq -e '
    .acceptance_mode == "canonical"
    and .overall_pass == true
    and .c1_gate_pass == true
    and .comparable_qos_gate_pass == true
    and .best_c1.profiled_errors == 0
    and .best_c1.post_profile_jit_events == 0
    and .best_c1.replay_rc == 0
    and .best_comparable_qos.profiled_errors == 0
    and .best_comparable_qos.post_profile_jit_events == 0
    and .best_comparable_qos.replay_rc == 0
' "$performance_report" >/dev/null

test -s "$eval_dir/run-metadata.txt"
test -s "$eval_dir/vllm_command.txt"
grep -qx 'indexer_kv_dtype=fp8' "$eval_dir/run-metadata.txt"
grep -qx 'target_attention_backend=ROCM_AITER_UNIFIED_ATTN' \
    "$eval_dir/run-metadata.txt"
grep -qx 'draft_attention_backend=ROCM_AITER_UNIFIED_ATTN' \
    "$eval_dir/run-metadata.txt"
grep -qx 'aiter_fused_cache_insert=1' "$eval_dir/run-metadata.txt"
grep -qx 'aiter_fused_ar_gemma=1' "$eval_dir/run-metadata.txt"
grep -q 'eagle3' "$eval_dir/vllm_command.txt"
if grep -q 'rejection_sample_method.*synthetic' "$eval_dir/vllm_command.txt"; then
    echo "FAIL: GSM8K used synthetic acceptance instead of target verification" >&2
    exit 1
fi

eval_result="$({
    python3 - "$eval_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for path in root.glob("*.json"):
    if path.name == "meta_env.json":
        continue
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if isinstance(data, dict) and "lm_eval_version" in data:
        candidates.append(path)
if not candidates:
    raise SystemExit("FAIL: no lm-eval result JSON found")
print(max(candidates, key=lambda path: path.stat().st_mtime))
PY
} 2>&1)" || {
    echo "$eval_result" >&2
    exit 1
}

jq -e '
    .["n-samples"].gsm8k.effective == 1319
    and (.results.gsm8k["exact_match,strict-match"] | type == "number")
    and (.results.gsm8k["exact_match,flexible-extract"] | type == "number")
' "$eval_result" >/dev/null

python3 "$inferencex_root/utils/evals/validate_scores.py" \
    --thresholds "$inferencex_root/utils/evals/thresholds.yaml" \
    --model-prefix minimaxm3 \
    --results-glob "$eval_result" \
    --meta-env "$eval_dir/meta_env.json"

printf 'PASS: canonical AgentX performance and full real-target GSM8K\n'
jq '{best_c1, best_comparable_qos, overall_pass}' "$performance_report"
jq '{results: .results.gsm8k, n_samples: .["n-samples"].gsm8k}' "$eval_result"
