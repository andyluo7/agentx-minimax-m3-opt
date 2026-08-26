#!/usr/bin/env bash
set -euo pipefail

readonly SHARED_ROOT="${AGENTX_SHARED_ROOT:-/shared/data/R7N/andy_luo_3v7/minimaxm3-agentx}"
readonly STORAGE_ROOT="$SHARED_ROOT/storage"
readonly LOCAL_RUNTIME_ROOT="/tmp/inferencex-minimaxm3-${SLURM_JOB_ID:-$$}"
readonly REPO_ROOT="$SHARED_ROOT/repos/InferenceX"
readonly DEFAULT_AGENTX_RECIPE="$REPO_ROOT/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh"
readonly EFFECTIVE_AGENTX_RECIPE="${AGENTX_RECIPE_OVERRIDE:-$DEFAULT_AGENTX_RECIPE}"
readonly CONTAINER_NAME="${AGENTX_CONTAINER_NAME:-minimaxm3-vllm-rocm-v0271}"
readonly ENROOT_RC="$SHARED_ROOT/enroot-command.rc"
readonly VLLM_MINIMAX_M3_SITE=/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3
readonly VLLM_SPECULATIVE_CONFIG_SITE=/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py
readonly VLLM_LLAMA_EAGLE3_SITE=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama_eagle3.py
readonly VLLM_LLAMA_SITE=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/llama.py
readonly VLLM_ATTENTION_LAYER_SITE=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py
readonly VLLM_AITER_UNIFIED_ATTN_SITE=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/rocm_aiter_unified_attn.py
readonly VLLM_MINIMAX_M3_WARMUP_SITE=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/minimax_m3_msa_warmup.py
readonly AITER_SPARSE_ATTN_SITE=/usr/local/lib/python3.12/dist-packages/aiter/ops/sparse_attention.py
readonly AITER_UNIFIED_ATTN_SITE=/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/attention/unified_attention.py
readonly AITER_UNIFIED_ATTN_KERNEL_SITE=/usr/local/lib/python3.12/dist-packages/aiter/ops/triton/_triton_kernels/attention/unified_attention.py
readonly AITER_FUSED_CACHE_INSERT_SITE=/usr/local/lib/python3.12/dist-packages/aiter/ops/minimax_m3_fused_qknorm_rope.py
readonly AITER_OPT_COMPILER_CONFIG_SITE=/usr/local/lib/python3.12/dist-packages/aiter/jit/optCompilerConfig.json

: "${CONC:?Set CONC to the AgentX concurrency.}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$CONC}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-512}"
export AITER_SPARSE_PRECOMPILE="${AITER_SPARSE_PRECOMPILE:-0}"
export INDEXER_KV_DTYPE="${INDEXER_KV_DTYPE:-fp8}"

if [[ ! -w "$SHARED_ROOT" ]]; then
    echo "Error: shared NFS root $SHARED_ROOT is absent or not writable." >&2
    exit 2
fi
if [[ ! -s "$EFFECTIVE_AGENTX_RECIPE" ]]; then
    echo "Error: AgentX recipe is missing or empty: $EFFECTIVE_AGENTX_RECIPE" >&2
    exit 2
fi

run_stamp="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_label="${RUN_LABEL:-resident-tp4-c${CONC}-maxseq${MAX_NUM_SEQS}-${run_stamp}}"
export RESULT_DIR="$SHARED_ROOT/results/$run_label"
readonly LOCAL_REPO_ROOT="$LOCAL_RUNTIME_ROOT/runs/$run_label/InferenceX"

mkdir -p \
    "$RESULT_DIR" \
    "$STORAGE_ROOT/models" \
    "$STORAGE_ROOT/hf" \
    "$LOCAL_RUNTIME_ROOT/home" \
    "$LOCAL_RUNTIME_ROOT/home/.cache" \
    "$LOCAL_RUNTIME_ROOT/home/.aiter/jit" \
    "$LOCAL_RUNTIME_ROOT/traces" \
    "$LOCAL_RUNTIME_ROOT/runs/$run_label" \
    "$LOCAL_REPO_ROOT" \
    "$LOCAL_RUNTIME_ROOT/uv-cache" \
    "$LOCAL_RUNTIME_ROOT/uv-bin" \
    "$LOCAL_RUNTIME_ROOT/tmp"

# Keep the long-running recipe and sourced helpers on node-local storage. A
# completed control previously received ESTALE from the shared NFS while bash
# was reading the recipe during final cleanup, after valid artifacts were saved.
rsync -a --exclude=.git "$REPO_ROOT/" "$LOCAL_REPO_ROOT/"
test -s "$LOCAL_REPO_ROOT/benchmarks/single_node/agentic/minimaxm3_fp4_mi355x_mtp.sh"

export HOME="$LOCAL_RUNTIME_ROOT/home"
export XDG_CACHE_HOME="$LOCAL_RUNTIME_ROOT/home/.cache"
export HF_HOME="$STORAGE_ROOT/hf"
export HF_HUB_CACHE="$STORAGE_ROOT/hf/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TMPDIR="$LOCAL_RUNTIME_ROOT/tmp"
export VLLM_CACHE_ROOT="$LOCAL_RUNTIME_ROOT/home/.cache/vllm"
export AIPERF_DATASET_MMAP_CACHE_DIR="$LOCAL_RUNTIME_ROOT/traces"
export AIPERF_RUNTIME_DIR="$LOCAL_RUNTIME_ROOT/runs/$run_label"
export AIPERF_VENV="$AIPERF_RUNTIME_DIR/venv"
export AIPERF_UV_CACHE_DIR="$LOCAL_RUNTIME_ROOT/uv-cache"
export AIPERF_UV_INSTALL_DIR="$LOCAL_RUNTIME_ROOT/uv-bin"

export MODEL=amd/MiniMax-M3-MXFP4
export MODEL_REVISION=b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a
export MODEL_PATH="$STORAGE_ROOT/models/amd--MiniMax-M3-MXFP4"
export DRAFT_MODEL_REVISION=96692486b5fd38ebf8fd2a5f6bb53427d30819a8
export DRAFT_MODEL_PATH="$STORAGE_ROOT/models/Inferact--MiniMax-M3-EAGLE3-GQA"
export MODEL_PREFIX=minimaxm3
export FRAMEWORK=vllm
export PRECISION=fp4
export IMAGE=vllm/vllm-openai-rocm:v0.27.1
export RUNNER_TYPE=cluster:mi355x-amds
export TP="${TP:-4}"
export PP_SIZE=1
export DCP_SIZE=1
export PCP_SIZE=1
export EP_SIZE=1
export DP_ATTENTION=false
export CONC
export PORT="${PORT:-8886}"
export KV_OFFLOADING="${KV_OFFLOADING:-none}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1547}"
export DURATION="${DURATION:-3600}"
export EVAL_ONLY="${EVAL_ONLY:-false}"
export RUN_EVAL="${RUN_EVAL:-false}"
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-3}"
case "$NUM_SPEC_TOKENS" in
    1) golden_synthetic_accept_len=1.79 ;;
    2) golden_synthetic_accept_len=2.39 ;;
    3) golden_synthetic_accept_len=2.78 ;;
    4) golden_synthetic_accept_len=3.02 ;;
    5) golden_synthetic_accept_len=3.21 ;;
    6) golden_synthetic_accept_len=3.30 ;;
    7) golden_synthetic_accept_len=3.52 ;;
    8) golden_synthetic_accept_len=3.51 ;;
    *)
        echo "Error: no committed MiniMax-M3 GQA golden acceptance for NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS." >&2
        exit 2
        ;;
esac
export SYNTHETIC_ACCEPT_LEN="${SYNTHETIC_ACCEPT_LEN:-$golden_synthetic_accept_len}"
if [[ "$EVAL_ONLY" != "true" \
    && "$SYNTHETIC_ACCEPT_LEN" != "$golden_synthetic_accept_len" \
    && "${ALLOW_NON_GOLDEN_ACCEPTANCE:-0}" != "1" ]]; then
    echo "Error: NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS requires the committed GQA golden acceptance length $golden_synthetic_accept_len, got $SYNTHETIC_ACCEPT_LEN." >&2
    echo "Set ALLOW_NON_GOLDEN_ACCEPTANCE=1 only for a non-submittable diagnostic." >&2
    exit 2
fi
if [[ "${DISABLE_SPECULATIVE_DECODING:-0}" == "1" ]]; then
    export SPEC_DECODING=none
else
    export SPEC_DECODING=mtp
fi
export DISAGG=false
export IS_MULTINODE=false
export SCENARIO_TYPE=agentic-coding
export IS_AGENTIC=1
export AIPERF_EXPERIMENTAL_FAST="${AIPERF_EXPERIMENTAL_FAST:-1}"
export AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-900}"
export AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT:-900000}"
export INFMAX_CONTAINER_WORKSPACE=/workspace
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$LOCAL_RUNTIME_ROOT/tmp/inferencex-pycache"
export RESULT_FILENAME="$run_label"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
export RECIPE_FINGERPRINT="${RECIPE_FINGERPRINT:-manual-node6}"
export REQUIRE_POWER="${REQUIRE_POWER:-0}"
export ENABLE_AGENTX_POWER="${ENABLE_AGENTX_POWER:-0}"

if [[ "$KV_OFFLOADING" == none ]]; then
    unset KV_OFFLOAD_BACKEND
    unset KV_OFFLOAD_BACKEND_METADATA
else
    export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:?Set KV_OFFLOAD_BACKEND for DRAM offload.}"
    export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"$KV_OFFLOAD_BACKEND\"}}"
fi

if [[ -n "${DRAFT_ATTENTION_BACKEND:-}" ]]; then
    effective_draft_attention_backend="$DRAFT_ATTENTION_BACKEND"
elif [[ "$KV_OFFLOADING" == none ]]; then
    effective_draft_attention_backend="ROCM_AITER_UNIFIED_ATTN"
else
    effective_draft_attention_backend="TRITON_ATTN"
fi

EXTRA_ENROOT_MOUNT_ARGS=()
if [[ -n "${VLLM_MINIMAX_M3_OVERLAY:-}" ]]; then
    if [[ ! -f "$VLLM_MINIMAX_M3_OVERLAY/common/indexer.py" \
        || ! -f "$VLLM_MINIMAX_M3_OVERLAY/amd/indexer_aiter.py" ]]; then
        echo "Error: incomplete vLLM MiniMax-M3 overlay: $VLLM_MINIMAX_M3_OVERLAY" >&2
        exit 2
    fi
    EXTRA_ENROOT_MOUNT_ARGS+=(
        --mount "$VLLM_MINIMAX_M3_OVERLAY:$VLLM_MINIMAX_M3_SITE"
    )
fi
if [[ -n "${VLLM_EAGLE3_OVERLAY:-}" ]]; then
    if [[ ! -f "$VLLM_EAGLE3_OVERLAY/vllm/config/speculative.py" \
        || ! -f "$VLLM_EAGLE3_OVERLAY/vllm/model_executor/models/llama_eagle3.py" \
        || ! -f "$VLLM_EAGLE3_OVERLAY/vllm/model_executor/models/llama.py" \
        || ! -f "$VLLM_EAGLE3_OVERLAY/vllm/model_executor/layers/attention/attention.py" \
        || ! -f "$VLLM_EAGLE3_OVERLAY/vllm/v1/attention/backends/rocm_aiter_unified_attn.py" ]]; then
        echo "Error: incomplete vLLM EAGLE3 overlay: $VLLM_EAGLE3_OVERLAY" >&2
        exit 2
    fi
    EXTRA_ENROOT_MOUNT_ARGS+=(
        --mount "$VLLM_EAGLE3_OVERLAY/vllm/config/speculative.py:$VLLM_SPECULATIVE_CONFIG_SITE"
        --mount "$VLLM_EAGLE3_OVERLAY/vllm/model_executor/models/llama_eagle3.py:$VLLM_LLAMA_EAGLE3_SITE"
        --mount "$VLLM_EAGLE3_OVERLAY/vllm/model_executor/models/llama.py:$VLLM_LLAMA_SITE"
        --mount "$VLLM_EAGLE3_OVERLAY/vllm/model_executor/layers/attention/attention.py:$VLLM_ATTENTION_LAYER_SITE"
        --mount "$VLLM_EAGLE3_OVERLAY/vllm/v1/attention/backends/rocm_aiter_unified_attn.py:$VLLM_AITER_UNIFIED_ATTN_SITE"
    )
fi
if [[ -n "${VLLM_AITER_UNIFIED_ATTN_OVERLAY:-}" ]]; then
    if [[ ! -f "$VLLM_AITER_UNIFIED_ATTN_OVERLAY/rocm_aiter_unified_attn.py" ]]; then
        echo "Error: incomplete AITER unified-attention vLLM overlay: $VLLM_AITER_UNIFIED_ATTN_OVERLAY" >&2
        exit 2
    fi
    EXTRA_ENROOT_MOUNT_ARGS+=(
        --mount "$VLLM_AITER_UNIFIED_ATTN_OVERLAY/rocm_aiter_unified_attn.py:$VLLM_AITER_UNIFIED_ATTN_SITE"
    )
fi
if [[ -n "${VLLM_MINIMAX_M3_WARMUP_OVERLAY:-}" ]]; then
    if [[ ! -f "$VLLM_MINIMAX_M3_WARMUP_OVERLAY/minimax_m3_msa_warmup.py" ]]; then
        echo "Error: incomplete MiniMax-M3 warmup overlay: $VLLM_MINIMAX_M3_WARMUP_OVERLAY" >&2
        exit 2
    fi
    EXTRA_ENROOT_MOUNT_ARGS+=(
        --mount "$VLLM_MINIMAX_M3_WARMUP_OVERLAY/minimax_m3_msa_warmup.py:$VLLM_MINIMAX_M3_WARMUP_SITE"
    )
fi
if [[ -n "${AITER_INDEXER_OVERLAY:-}" ]]; then
    if [[ ! -f "$AITER_INDEXER_OVERLAY/aiter/ops/sparse_attention.py" \
        || ! -f "$AITER_INDEXER_OVERLAY/csrc/cpp_itfs/sparse_attn/pa_sparse_block_select.py" ]]; then
        echo "Error: incomplete AITER indexer overlay: $AITER_INDEXER_OVERLAY" >&2
        exit 2
    fi
    export PYTHONPATH="$AITER_INDEXER_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
    EXTRA_ENROOT_MOUNT_ARGS+=(
        --mount "$AITER_INDEXER_OVERLAY/aiter/ops/sparse_attention.py:$AITER_SPARSE_ATTN_SITE"
    )
    if [[ -f "$AITER_INDEXER_OVERLAY/aiter/ops/triton/attention/unified_attention.py" ]]; then
        if [[ ! -f "$AITER_INDEXER_OVERLAY/aiter/ops/triton/_triton_kernels/attention/unified_attention.py" ]]; then
            echo "Error: incomplete AITER unified-attention overlay: $AITER_INDEXER_OVERLAY" >&2
            exit 2
        fi
        EXTRA_ENROOT_MOUNT_ARGS+=(
            --mount "$AITER_INDEXER_OVERLAY/aiter/ops/triton/attention/unified_attention.py:$AITER_UNIFIED_ATTN_SITE"
            --mount "$AITER_INDEXER_OVERLAY/aiter/ops/triton/_triton_kernels/attention/unified_attention.py:$AITER_UNIFIED_ATTN_KERNEL_SITE"
        )
    fi
    if [[ -f "$AITER_INDEXER_OVERLAY/aiter/ops/minimax_m3_fused_qknorm_rope.py" ]]; then
        if [[ ! -f "$AITER_INDEXER_OVERLAY/aiter/jit/optCompilerConfig.json" \
            || ! -f "$AITER_INDEXER_OVERLAY/csrc/kernels/minimax_m3_fused_qknorm_rope_cache_shuffle.cu" ]]; then
            echo "Error: incomplete AITER fused cache-insert overlay: $AITER_INDEXER_OVERLAY" >&2
            exit 2
        fi
        export AITER_FUSED_CACHE_INSERT_OVERLAY="$AITER_INDEXER_OVERLAY"
        export AITER_JIT_DIR="$LOCAL_RUNTIME_ROOT/home/.aiter/jit"
        EXTRA_ENROOT_MOUNT_ARGS+=(
            --mount "$AITER_INDEXER_OVERLAY/aiter/ops/minimax_m3_fused_qknorm_rope.py:$AITER_FUSED_CACHE_INSERT_SITE"
            --mount "$AITER_INDEXER_OVERLAY/aiter/jit/optCompilerConfig.json:$AITER_OPT_COMPILER_CONFIG_SITE"
        )
    fi
fi
if [[ "${VLLM_MINIMAX_M3_FUSED_CACHE_INSERT:-0}" == "1" ]]; then
    if [[ -z "${VLLM_MINIMAX_M3_OVERLAY:-}" \
        || ! -f "$VLLM_MINIMAX_M3_OVERLAY/amd/model.py" \
        || -z "${AITER_FUSED_CACHE_INSERT_OVERLAY:-}" ]]; then
        echo "Error: fused cache insert requires the combined vLLM and AITER overlays." >&2
        exit 2
    fi
fi
if [[ "${VLLM_MINIMAX_M3_ASM_SPARSE_ATTN:-0}" != "0" \
    && "${VLLM_MINIMAX_M3_ASM_SPARSE_ATTN:-0}" != "1" ]]; then
    echo "Error: VLLM_MINIMAX_M3_ASM_SPARSE_ATTN must be 0 or 1." >&2
    exit 2
fi
if [[ "${AITER_UNIFIED_ATTN_SLIDING_DECODE_3D:-0}" != "0" \
    && "${AITER_UNIFIED_ATTN_SLIDING_DECODE_3D:-0}" != "1" ]]; then
    echo "Error: AITER_UNIFIED_ATTN_SLIDING_DECODE_3D must be 0 or 1." >&2
    exit 2
fi

effective_profile_duration="$DURATION"
effective_warmup_requests_per_lane="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
if [[ "$AIPERF_EXPERIMENTAL_FAST" == "1" ]]; then
    effective_profile_duration=1200
    effective_warmup_requests_per_lane=1
fi

{
    printf 'run_label=%s\n' "$run_label"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
    printf 'repo_sha=%s\n' "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf 'recipe_sha256=%s\n' "$(
        sha256sum "$EFFECTIVE_AGENTX_RECIPE" | cut -d' ' -f1
    )"
    printf 'recipe_path=%s\n' "$EFFECTIVE_AGENTX_RECIPE"
    printf 'lm_eval_sitecustomize_sha256=%s\n' "$(
        sha256sum \
            "$REPO_ROOT/utils/evals/patches/lm_eval_sitecustomize.py" \
            | cut -d' ' -f1
    )"
    printf 'runner_sha256=%s\n' "$(
        sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1
    )"
    printf 'container_runner_sha256=%s\n' "$(
        sha256sum "$SHARED_ROOT/container-run-agentx.sh" | cut -d' ' -f1
    )"
    printf 'container_registry_digest=%s\n' 'sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7'
    printf 'container_name=%s\n' "$CONTAINER_NAME"
    printf 'model_revision=%s\n' "$MODEL_REVISION"
    printf 'draft_model_revision=%s\n' "$DRAFT_MODEL_REVISION"
    printf 'conc=%s\n' "$CONC"
    printf 'max_num_seqs=%s\n' "$MAX_NUM_SEQS"
    printf 'max_num_batched_tokens=%s\n' "$MAX_NUM_BATCHED_TOKENS"
    printf 'gpu_memory_utilization=%s\n' "$GPU_MEMORY_UTILIZATION"
    printf 'max_cudagraph_capture_size=%s\n' "$MAX_CUDAGRAPH_CAPTURE_SIZE"
    printf 'fast_mode=%s\n' "$AIPERF_EXPERIMENTAL_FAST"
    printf 'configured_duration_seconds=%s\n' "$DURATION"
    printf 'eval_only=%s\n' "$EVAL_ONLY"
    printf 'eval_limit=%s\n' "${EVAL_LIMIT:-full}"
    printf 'duration_seconds=%s\n' "$effective_profile_duration"
    printf 'warmup_requests_per_lane=%s\n' "$effective_warmup_requests_per_lane"
    printf 'kernel_prime_duration_seconds=%s\n' \
        "${AIPERF_KERNEL_PRIME_DURATION_SECONDS:-0}"
    printf 'vllm_http_timeout_keep_alive=%s\n' "$VLLM_HTTP_TIMEOUT_KEEP_ALIVE"
    printf 'aiperf_http_tcp_user_timeout=%s\n' "$AIPERF_HTTP_TCP_USER_TIMEOUT"
    printf 'kv_offloading=%s\n' "$KV_OFFLOADING"
    printf 'kv_offload_backend=%s\n' "${KV_OFFLOAD_BACKEND:-none}"
    printf 'model_storage=%s\n' 'shared-nfs'
    printf 'runtime_cache=%s\n' 'node-local-/tmp'
    printf 'speculative_decoding=%s\n' "$([[ "${DISABLE_SPECULATIVE_DECODING:-0}" == "1" ]] && printf disabled || printf eagle3)"
    printf 'num_speculative_tokens=%s\n' "$NUM_SPEC_TOKENS"
    printf 'synthetic_acceptance_length=%s\n' "$SYNTHETIC_ACCEPT_LEN"
    printf 'golden_synthetic_acceptance_length=%s\n' \
        "$golden_synthetic_accept_len"
    printf 'golden_synthetic_acceptance_source=%s\n' \
        'golden_al_distribution/minimaxm3_eagle3_gqa.yaml'
    printf 'non_golden_acceptance_override=%s\n' \
        "${ALLOW_NON_GOLDEN_ACCEPTANCE:-0}"
    printf 'target_attention_backend=%s\n' "${TARGET_ATTENTION_BACKEND:-TRITON_ATTN}"
    printf 'draft_attention_backend=%s\n' "$effective_draft_attention_backend"
    printf 'draft_max_model_len=%s\n' "${DRAFT_MAX_MODEL_LEN:-target-default}"
    printf 'draft_attention_window=%s\n' "${DRAFT_ATTENTION_WINDOW:-full-context}"
    printf 'aiter_sliding_decode_3d=%s\n' "${AITER_UNIFIED_ATTN_SLIDING_DECODE_3D:-0}"
    printf 'draft_use_local_argmax_reduction=%s\n' "${DRAFT_USE_LOCAL_ARGMAX_REDUCTION:-0}"
    printf 'indexer_kv_dtype=%s\n' "${INDEXER_KV_DTYPE:-bf16}"
    printf 'target_kv_cache_dtype=%s\n' "${KV_CACHE_DTYPE:-fp8}"
    printf 'breakable_cudagraph=%s\n' "${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
    printf 'vllm_minimax_m3_overlay=%s\n' "${VLLM_MINIMAX_M3_OVERLAY:-none}"
    printf 'vllm_eagle3_overlay=%s\n' "${VLLM_EAGLE3_OVERLAY:-none}"
    printf 'vllm_aiter_unified_attn_overlay=%s\n' \
        "${VLLM_AITER_UNIFIED_ATTN_OVERLAY:-none}"
    if [[ -n "${VLLM_AITER_UNIFIED_ATTN_OVERLAY:-}" ]]; then
        printf 'vllm_aiter_unified_attn_sha256=%s\n' "$(
            sha256sum \
                "${VLLM_AITER_UNIFIED_ATTN_OVERLAY}/rocm_aiter_unified_attn.py" \
                | cut -d' ' -f1
        )"
    else
        printf 'vllm_aiter_unified_attn_sha256=none\n'
    fi
    printf 'aiter_unified_attn_quant_query=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY:-1}"
    printf 'aiter_unified_attn_quant_output=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT:-1}"
    printf 'aiter_unified_attn_kernel=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL:-aiter}"
    printf 'aiter_unified_attn_cache_writer=%s\n' \
        "${VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER:-aiter}"
    printf 'aiter_indexer_overlay=%s\n' "${AITER_INDEXER_OVERLAY:-none}"
    printf 'aiter_fused_cache_insert_overlay=%s\n' "${AITER_FUSED_CACHE_INSERT_OVERLAY:-none}"
    printf 'aiter_fused_cache_insert=%s\n' "${VLLM_MINIMAX_M3_FUSED_CACHE_INSERT:-0}"
    printf 'aiter_fused_ar_gemma=%s\n' "${VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA:-0}"
    printf 'rocm_fp32_router_gemm=%s\n' "${VLLM_MINIMAX_M3_ROCM_FP32_ROUTER_GEMM:-0}"
    printf 'agentx_jit_warmup=%s\n' "${VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP:-0}"
    printf 'vllm_server_dev_mode=%s\n' "${VLLM_SERVER_DEV_MODE:-0}"
    printf 'agentx_jit_warmup_overlay=%s\n' "${VLLM_MINIMAX_M3_WARMUP_OVERLAY:-none}"
    if [[ -n "${VLLM_MINIMAX_M3_WARMUP_OVERLAY:-}" \
        && -f "$VLLM_MINIMAX_M3_WARMUP_OVERLAY/minimax_m3_msa_warmup.py" ]]; then
        printf 'agentx_jit_warmup_sha256=%s\n' "$(
            sha256sum \
                "${VLLM_MINIMAX_M3_WARMUP_OVERLAY}/minimax_m3_msa_warmup.py" \
                | cut -d' ' -f1
        )"
    else
        printf 'agentx_jit_warmup_sha256=none\n'
    fi
    printf 'jit_monitor_verbose=%s\n' "${JIT_MONITOR_VERBOSE:-1}"
    printf 'aiter_asm_sparse_attention=%s\n' "${VLLM_MINIMAX_M3_ASM_SPARSE_ATTN:-0}"
    printf 'aiter_sparse_precompile=%s\n' "$AITER_SPARSE_PRECOMPILE"
    printf 'quick_reduce_max_size_mb=%s\n' "${VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB:-2048}"
    printf 'aiter_config_gemm_bf16=%s\n' "${AITER_CONFIG_GEMM_BF16:-auto}"
    printf 'aiter_config_fmoe=%s\n' "${AITER_CONFIG_FMOE:-auto}"
    printf 'agentx_power=%s\n' "$ENABLE_AGENTX_POWER"
} | tee "$RESULT_DIR/run-metadata.txt"

ENROOT_ENV_KEYS=(
    AGENTX_SHARED_ROOT HOME XDG_CACHE_HOME HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
    TRANSFORMERS_CACHE TMPDIR VLLM_CACHE_ROOT AIPERF_DATASET_MMAP_CACHE_DIR
    AIPERF_RUNTIME_DIR AIPERF_VENV AIPERF_UV_CACHE_DIR AIPERF_UV_INSTALL_DIR
    MODEL MODEL_REVISION MODEL_PATH DRAFT_MODEL_REVISION DRAFT_MODEL_PATH
    MODEL_PREFIX FRAMEWORK PRECISION IMAGE RUNNER_TYPE TP PP_SIZE DCP_SIZE
    PCP_SIZE EP_SIZE DP_ATTENTION CONC PORT RESULT_DIR RESULT_FILENAME
    AGENTIC_OUTPUT_DIR RECIPE_FINGERPRINT REQUIRE_POWER ENABLE_AGENTX_POWER
    KV_OFFLOADING TOTAL_CPU_DRAM_GB DURATION EVAL_ONLY RUN_EVAL
    SPEC_DECODING DISAGG IS_MULTINODE SCENARIO_TYPE IS_AGENTIC
    AIPERF_EXPERIMENTAL_FAST
    AIPERF_FAILED_REQUEST_THRESHOLD AIPERF_HTTP_TCP_USER_TIMEOUT
    VLLM_HTTP_TIMEOUT_KEEP_ALIVE INFMAX_CONTAINER_WORKSPACE
    PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX MAX_NUM_SEQS
    MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION
)
for optional_key in \
    KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA MAX_CUDAGRAPH_CAPTURE_SIZE \
    DISABLE_SPECULATIVE_DECODING TARGET_ATTENTION_BACKEND DRAFT_ATTENTION_BACKEND \
    VLLM_AITER_UNIFIED_ATTN_OVERLAY \
    VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_QUERY \
    VLLM_ROCM_AITER_UNIFIED_ATTN_QUANT_OUTPUT \
    VLLM_ROCM_AITER_UNIFIED_ATTN_KERNEL \
    VLLM_ROCM_AITER_UNIFIED_ATTN_CACHE_WRITER \
    DRAFT_MAX_MODEL_LEN DRAFT_ATTENTION_WINDOW DRAFT_USE_LOCAL_ARGMAX_REDUCTION \
    AITER_UNIFIED_ATTN_SLIDING_DECODE_3D \
    NUM_SPEC_TOKENS SYNTHETIC_ACCEPT_LEN ALLOW_NON_GOLDEN_ACCEPTANCE \
    EVAL_LIMIT EVAL_TASKS_DIR EVAL_MAX_MODEL_LEN EVAL_CONCURRENT_REQUESTS \
    AITER_CONFIG_GEMM_BF16 \
    AITER_CONFIG_FMOE \
    INDEXER_KV_DTYPE KV_CACHE_DTYPE VLLM_USE_BREAKABLE_CUDAGRAPH \
    AITER_SPARSE_PRECOMPILE MAX_MODEL_LEN \
    VLLM_MINIMAX_M3_FUSED_CACHE_INSERT PYTHONPATH \
    VLLM_MINIMAX_M3_AITER_FUSED_AR_GEMMA \
    VLLM_MINIMAX_M3_ROCM_FP32_ROUTER_GEMM \
    VLLM_MINIMAX_M3_AGENTX_JIT_WARMUP VLLM_MINIMAX_M3_WARMUP_OVERLAY \
    JIT_MONITOR_VERBOSE \
    VLLM_MINIMAX_M3_ASM_SPARSE_ATTN \
    AITER_FUSED_CACHE_INSERT_OVERLAY AITER_JIT_DIR \
    VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB \
    VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT WEKA_LOADER_OVERRIDE \
    AIPERF_WARMUP_REQUESTS_PER_LANE AIPERF_PYTHON_VERSION \
    AIPERF_KERNEL_PRIME_DURATION_SECONDS AGENTX_RECIPE_OVERRIDE \
    SLURM_JOB_ID SLURMD_NODENAME ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES \
    CUDA_VISIBLE_DEVICES; do
    if [[ -n "${!optional_key:-}" ]]; then
        ENROOT_ENV_KEYS+=("$optional_key")
    fi
done
ENROOT_ENV_ARGS=()
for env_key in "${ENROOT_ENV_KEYS[@]}"; do
    ENROOT_ENV_ARGS+=(--env "$env_key")
done

exec enroot start \
    --root \
    --rw \
    --rc "$ENROOT_RC" \
    "${ENROOT_ENV_ARGS[@]}" \
    --mount "$LOCAL_REPO_ROOT:/workspace" \
    --mount "$SHARED_ROOT:$SHARED_ROOT" \
    --mount "$LOCAL_RUNTIME_ROOT:$LOCAL_RUNTIME_ROOT" \
    --mount /dev/kfd:/dev/kfd \
    --mount /dev/dri:/dev/dri \
    "${EXTRA_ENROOT_MOUNT_ARGS[@]}" \
    "$CONTAINER_NAME" \
    bash "$SHARED_ROOT/container-run-agentx.sh" \
    2>&1 | tee "$RESULT_DIR/console.log"
