#!/usr/bin/env bash
set -euo pipefail

readonly shared_root=/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx
readonly model_path="$shared_root/storage/models/amd--MiniMax-M3-MXFP4"
readonly draft_path="$shared_root/storage/models/Inferact--MiniMax-M3-EAGLE3-GQA"
readonly profiler_dir="$RESULT_DIR/torch-profiler"
readonly server_log="$RESULT_DIR/server.log"
readonly target_attention_backend="${TARGET_ATTENTION_BACKEND:-ROCM_AITER_UNIFIED_ATTN}"
readonly draft_attention_backend="${DRAFT_ATTENTION_BACKEND:-ROCM_AITER_UNIFIED_ATTN}"
readonly profile_synthetic_acceptance="${PROFILE_SYNTHETIC_ACCEPTANCE:-0}"

if [[ "$profile_synthetic_acceptance" != "0" \
    && "$profile_synthetic_acceptance" != "1" ]]; then
    echo "Error: PROFILE_SYNTHETIC_ACCEPTANCE must be 0 or 1." >&2
    exit 2
fi

mkdir -p "$RESULT_DIR" "$profiler_dir"

speculative_args=()
if [[ "$PROFILE_MODE" != target-only* ]]; then
    draft_attention_window_field=""
    synthetic_acceptance_fields=""
    if [[ -n "${DRAFT_ATTENTION_WINDOW:-}" ]]; then
        draft_attention_window_field=$(printf \
            ',"draft_attention_window":%d' "$DRAFT_ATTENTION_WINDOW")
    fi
    if [[ "$profile_synthetic_acceptance" == "1" ]]; then
        synthetic_acceptance_fields=',"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.78'
    fi
    speculative_config=$(printf \
        '{"method":"eagle3","model":"%s","num_speculative_tokens":3,"attention_backend":"%s"%s%s}' \
        "$draft_path" "$draft_attention_backend" \
        "$draft_attention_window_field" "$synthetic_acceptance_fields")
    speculative_args=(--speculative-config "$speculative_config")
fi

profiler_config=$(printf \
    '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"torch_profiler_use_gzip":true,"ignore_frontend":true,"delay_iterations":2,"max_iterations":12}' \
    "$profiler_dir")

server_cmd=(
    vllm serve "$model_path"
    --served-model-name amd/MiniMax-M3-MXFP4
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tensor-parallel-size 4
    --block-size 128
    --gpu-memory-utilization 0.85
    --enable-chunked-prefill
    --max-num-batched-tokens 32768
    --language-model-only
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --attention-backend "$target_attention_backend"
    --attention-config '{"indexer_kv_dtype":"fp8"}'
    --moe-backend aiter
    --kv-cache-dtype fp8
    --tool-call-parser minimax_m3
    --reasoning-parser minimax_m3
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"thinking_mode":"enabled"}'
    --max-num-seqs 256
    --max-cudagraph-capture-size 8
    --jit-monitor-verbose
    --stream-interval 20
    --hf-overrides '{"text_config":{"use_index_cache":true,"index_topk_freq":4}}'
    --profiler-config "$profiler_config"
    "${speculative_args[@]}"
)

if [[ "$profile_synthetic_acceptance" == "1" ]]; then
    profile_acceptance=synthetic
else
    profile_acceptance=real-target
fi
{
    printf 'hostname=%s\n' "$(hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
    printf 'profile_mode=%s\n' "$PROFILE_MODE"
    printf 'profile_acceptance=%s\n' "$profile_acceptance"
    printf 'target_attention_backend=%s\n' "$target_attention_backend"
    printf 'draft_attention_backend=%s\n' "$draft_attention_backend"
    printf 'vllm_aiter_unified_attn_overlay=%s\n' \
        "${VLLM_AITER_UNIFIED_ATTN_OVERLAY:-none}"
    printf 'aiter_unified_attn_quant_query=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY:-1}"
    printf 'aiter_unified_attn_quant_output=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT:-1}"
    printf 'aiter_unified_attn_kernel=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL:-aiter}"
    printf 'aiter_unified_attn_cache_writer=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER:-aiter}"
    printf 'draft_attention_window=%s\n' "${DRAFT_ATTENTION_WINDOW:-full-context}"
    printf 'indexer_kv_dtype=fp8\n'
    printf 'aiter_sliding_decode_3d=%s\n' \
        "${AITER_UNIFIED_ATTN_SLIDING_DECODE_3D:-0}"
    printf 'aiter_fused_cache_insert=%s\n' \
        "${VLLM_MINIMAX_M3_FUSED_CACHE_INSERT:-0}"
    printf 'aiter_fused_ar_gemma=%s\n' \
        "${VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA:-0}"
    printf 'agentx_jit_warmup=%s\n' \
        "${VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP:-0}"
    printf 'agentx_jit_warmup_overlay=%s\n' \
        "${VLLM_MINIMAX_M3_WARMUP_OVERLAY:-none}"
} | tee "$RESULT_DIR/profile-metadata.txt"

printf '%q ' "${server_cmd[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"

"${server_cmd[@]}" >"$server_log" 2>&1 &
server_pid=$!
cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 360); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 200 "$server_log" >&2
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
        ready=1
        break
    fi
    sleep 10
done
if [[ "$ready" != 1 ]]; then
    echo "Error: server did not become healthy" >&2
    exit 1
fi

python3 "$shared_root/profile_minimaxm3_request.py"
curl -fsS "http://127.0.0.1:$PORT/metrics" >"$RESULT_DIR/server-metrics.prom"

cleanup
trap - EXIT INT TERM

test -s "$RESULT_DIR/profile-request-summary.json"
if ! find "$profiler_dir" -type f -size +0c -print -quit | grep -q .; then
    echo "Error: profiler produced no trace files" >&2
    exit 1
fi
