#!/usr/bin/env bash
set -euo pipefail

readonly SHARED_ROOT="${AGENTX_SHARED_ROOT:-/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx}"
readonly DEFAULT_AGENTX_RECIPE=/workspace/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh
readonly EFFECTIVE_AGENTX_RECIPE="${AGENTX_RECIPE_OVERRIDE:-$DEFAULT_AGENTX_RECIPE}"

if [[ ! -s "$EFFECTIVE_AGENTX_RECIPE" ]]; then
    echo "ERROR: AgentX recipe is missing or empty: $EFFECTIVE_AGENTX_RECIPE" >&2
    exit 2
fi

if [[ -n "${VLLM_SOURCE_ROOT:-}" ]]; then
    if [[ ! -f "$VLLM_SOURCE_ROOT/vllm/__init__.py" ]]; then
        echo "ERROR: invalid vLLM source root: $VLLM_SOURCE_ROOT" >&2
        exit 2
    fi
    if [[ -n "${VLLM_INTEGRATION_SOURCE_SHA:-}" ]] \
        && [[ "$(git -C "$VLLM_SOURCE_ROOT" rev-parse HEAD)" \
            != "$VLLM_INTEGRATION_SOURCE_SHA" ]]; then
        echo "ERROR: vLLM source root does not match the pinned integration SHA" >&2
        exit 2
    fi

    readonly installed_vllm=/usr/local/lib/python3.12/dist-packages/vllm
    readonly source_overlay="$TMPDIR/vllm-source-overlay-${SLURM_JOB_ID:-$$}"
    if [[ -e "$source_overlay" ]]; then
        echo "ERROR: vLLM source overlay already exists: $source_overlay" >&2
        exit 2
    fi
    mkdir -p "$source_overlay/vllm"
    rsync -a "$VLLM_SOURCE_ROOT/vllm/" "$source_overlay/vllm/"

    # The PR stack is Python-only. Reuse the ABI-matched extensions from the
    # pinned official image while importing every Python module from the exact
    # integration checkout. Preserve relative paths for nested extension
    # modules such as vllm_flash_attn.
    while IFS= read -r -d '' extension; do
        relative_path="${extension#"$installed_vllm/"}"
        install -D -m 0644 "$extension" "$source_overlay/vllm/$relative_path"
    done < <(find "$installed_vllm" -type f -name '*.so' -print0)
    if [[ -f "$installed_vllm/_version.py" ]]; then
        install -m 0644 "$installed_vllm/_version.py" \
            "$source_overlay/vllm/_version.py"
    fi

    export PYTHONPATH="$source_overlay${PYTHONPATH:+:$PYTHONPATH}"
    python - <<'PY'
import vllm

print(f"current_head_vllm_module={vllm.__file__}")
PY
fi

if [[ "${AITER_SPARSE_PRECOMPILE:-0}" == "1" ]]; then
    python3 "$SHARED_ROOT/precompile_aiter_sparse.py" \
        --max-model-len "${MAX_MODEL_LEN:-1048576}"
fi

# append_lm_eval_summary stages its JSON artifacts into the caller's current
# directory. Keep that directory on shared storage so eval evidence survives
# destruction of the node-local Enroot runtime.
cd "$RESULT_DIR"

recipe_rc=0
bash "$EFFECTIVE_AGENTX_RECIPE" \
    || recipe_rc=$?

if [[ "$recipe_rc" -eq 0 && "${EVAL_ONLY:-false}" == "true" ]]; then
    result_files=("$RESULT_DIR"/results*.json)
    if [[ ! -e "${result_files[0]}" ]]; then
        echo "ERROR: eval completed without a persistent results*.json artifact" >&2
        recipe_rc=1
    else
        python3 /workspace/utils/evals/validate_scores.py \
            --thresholds /workspace/utils/evals/thresholds.yaml \
            --model-prefix minimaxm3 \
            --results-glob "$RESULT_DIR/results*.json" \
            --meta-env "$RESULT_DIR/meta_env.json" \
            || recipe_rc=$?
    fi
fi

exit "$recipe_rc"
