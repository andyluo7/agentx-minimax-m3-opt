#!/usr/bin/env python3
"""Production-shape MI355X microbenchmarks for the MiniMax-M3 latency stack.

The indexer comparison includes the page-table conversion consumed by sparse
attention: Triton BF16 score/top-k followed by the separate table builder versus
AITER FP8 score/top-k with the table emitted by the top-k kernel.  The cache
comparison includes the same three operations used by the model before and
after the fused cache-insert patch.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable

import torch
from aiter import reshape_and_cache
from aiter.ops.minimax_m3_fused_qknorm_rope import (
    minimax_m3_qknorm_rope_cache_shuffle_insert,
)
from aiter.ops.sparse_attention import (
    pa_sparse_block_score_decode,
    pa_sparse_block_topk,
)
from vllm import _custom_ops as ops
from vllm.models.minimax_m3.amd.ops.index_topk import minimax_m3_index_decode
from vllm.models.minimax_m3.amd.ops.sparse_pa import (
    minimax_m3_build_sparse_block_table_decode,
    minimax_m3_insert_index_cache,
)


SPARSE_BLOCK_SIZE = 128
MAIN_PAGE_SIZE = 16
HEAD_DIM = 128
TOPK = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
SPARSE_LAYERS = 57


def score_width(seq_len: int) -> int:
    blocks = math.ceil(seq_len / SPARSE_BLOCK_SIZE)
    strips = math.ceil(blocks / 64)
    return (1 << (strips - 1).bit_length()) * 64


def elapsed_samples_ms(
    fn: Callable[[], None], *, warmup: int = 8, repeats: int = 50
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def indexer_case(seq_len: int, query_len: int) -> dict[str, object]:
    device = torch.device("cuda")
    num_sparse_blocks = math.ceil(seq_len / SPARSE_BLOCK_SIZE)
    num_main_pages = math.ceil(seq_len / MAIN_PAGE_SIZE)

    torch.manual_seed(17 + seq_len + query_len)
    query_bf16 = torch.randn(
        query_len, 1, HEAD_DIM, device=device, dtype=torch.bfloat16
    )
    cache_bf16 = torch.randn(
        num_sparse_blocks,
        SPARSE_BLOCK_SIZE,
        HEAD_DIM,
        device=device,
        dtype=torch.bfloat16,
    )
    query_fp8 = query_bf16.to(torch.float8_e4m3fn)
    cache_fp8 = cache_bf16.to(torch.float8_e4m3fn)
    index_block_table = torch.arange(
        num_sparse_blocks, device=device, dtype=torch.int32
    )[None]
    main_block_table = torch.arange(
        num_main_pages, device=device, dtype=torch.int32
    )[None]
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

    triton_topk = torch.empty(
        (1, query_len, TOPK), device=device, dtype=torch.int32
    )
    aiter_score = torch.empty(
        (1, query_len, score_width(seq_len)), device=device, dtype=torch.float32
    )
    aiter_topk = torch.empty_like(triton_topk)
    aiter_sparse_bt = torch.empty(
        (query_len, TOPK * (SPARSE_BLOCK_SIZE // MAIN_PAGE_SIZE)),
        device=device,
        dtype=torch.int32,
    )
    aiter_sparse_ctx = torch.empty(query_len, device=device, dtype=torch.int32)

    def triton_path() -> None:
        topk = minimax_m3_index_decode(
            query_bf16,
            cache_bf16,
            index_block_table,
            seq_lens,
            seq_len,
            TOPK,
            INIT_BLOCKS,
            LOCAL_BLOCKS,
            1,
            query_len,
            query_len,
            out=triton_topk,
        )
        minimax_m3_build_sparse_block_table_decode(
            topk, main_block_table, seq_lens, query_len
        )

    def aiter_path() -> None:
        pa_sparse_block_score_decode(
            query_fp8,
            cache_fp8,
            aiter_score,
            index_block_table,
            seq_lens,
            init_blocks=INIT_BLOCKS,
            local_blocks=LOCAL_BLOCKS,
            query_len=query_len,
            max_seq_len=seq_len,
        )
        pa_sparse_block_topk(
            aiter_score,
            aiter_topk,
            main_block_table,
            seq_lens,
            aiter_sparse_bt,
            aiter_sparse_ctx,
            max_seq_len=seq_len,
            block_size=SPARSE_BLOCK_SIZE,
            query_len=query_len,
            num_kv_heads=1,
            pages_per_block=SPARSE_BLOCK_SIZE // MAIN_PAGE_SIZE,
        )

    triton_ms = elapsed_samples_ms(triton_path)
    aiter_ms = elapsed_samples_ms(aiter_path)
    triton_summary = summarize(triton_ms)
    aiter_summary = summarize(aiter_ms)
    per_layer_delta = triton_summary["p50_ms"] - aiter_summary["p50_ms"]
    return {
        "seq_len": seq_len,
        "query_len": query_len,
        "triton_bf16_plus_table": triton_summary,
        "aiter_fp8_emitted_table": aiter_summary,
        "speedup_x_p50": triton_summary["p50_ms"] / aiter_summary["p50_ms"],
        "projected_57_layer_p50_savings_ms": per_layer_delta * SPARSE_LAYERS,
    }


def allocate_caches(
    num_pages: int, num_kv_heads: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = 16 // torch.empty((), dtype=dtype).element_size()
    device = torch.device("cuda")
    key = torch.zeros(
        num_pages,
        num_kv_heads,
        HEAD_DIM // x,
        MAIN_PAGE_SIZE,
        x,
        device=device,
        dtype=dtype,
    )
    value = torch.zeros(
        num_pages,
        num_kv_heads,
        MAIN_PAGE_SIZE // x,
        HEAD_DIM,
        x,
        device=device,
        dtype=dtype,
    )
    index = torch.zeros(
        num_pages,
        MAIN_PAGE_SIZE,
        HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    return key, value, index


def cache_insert_case(num_tokens: int) -> dict[str, object]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache_dtype = torch.float8_e4m3fn
    num_heads = 16
    num_kv_heads = 1
    num_index_heads = 1
    rotary_dim = 64
    eps = 1e-6
    num_pages = 32
    row_width = (num_heads + 2 * num_kv_heads + num_index_heads + 1) * HEAD_DIM

    torch.manual_seed(29 + num_tokens)
    qkv = torch.randn(num_tokens, row_width, device=device, dtype=dtype)
    weights = [
        torch.randn(HEAD_DIM, device=device, dtype=dtype) * 0.1 for _ in range(4)
    ]
    cos_sin = torch.randn(8192, rotary_dim, device=device, dtype=dtype)
    positions = torch.arange(4096, 4096 + num_tokens, device=device)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int64) + 15
    index_slot_mapping = (
        torch.arange(num_tokens, device=device, dtype=torch.int64) + 31
    )
    k_scale = torch.tensor([1.0], device=device)
    v_scale = torch.tensor([1.0], device=device)
    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    index_q_size = num_index_heads * HEAD_DIM

    baseline_qkv = qkv.clone()
    baseline_q = torch.empty(num_tokens, q_size, device=device, dtype=dtype)
    baseline_index_q = torch.empty(
        num_tokens, index_q_size, device=device, dtype=cache_dtype
    )
    baseline_k, baseline_v, baseline_index = allocate_caches(
        num_pages, num_kv_heads, cache_dtype
    )
    fused_q = torch.empty_like(baseline_q)
    fused_index_q = torch.empty_like(baseline_index_q)
    fused_k, fused_v, fused_index = allocate_caches(
        num_pages, num_kv_heads, cache_dtype
    )

    def baseline_path() -> None:
        ops.fused_minimax_m3_qknorm_rope_kv_insert(
            baseline_qkv,
            weights[0],
            weights[1],
            cos_sin,
            positions,
            num_heads,
            num_kv_heads,
            rotary_dim,
            eps,
            weights[2],
            weights[3],
            num_index_heads,
            q_out=baseline_q,
            index_q_out=baseline_index_q,
            kv_cache_dtype="fp8",
        )
        k_start = q_size
        v_start = k_start + kv_size
        index_k_start = v_start + kv_size + index_q_size
        key = baseline_qkv[:, k_start:v_start].view(
            num_tokens, num_kv_heads, HEAD_DIM
        )
        value = baseline_qkv[:, v_start : v_start + kv_size].view(
            num_tokens, num_kv_heads, HEAD_DIM
        )
        index_key = baseline_qkv[:, index_k_start : index_k_start + HEAD_DIM]
        reshape_and_cache(
            key.contiguous(),
            value.contiguous(),
            baseline_k,
            baseline_v,
            slot_mapping,
            kv_cache_dtype="fp8",
            k_scale=k_scale,
            v_scale=v_scale,
            asm_layout=True,
        )
        minimax_m3_insert_index_cache(
            index_key, baseline_index, index_slot_mapping
        )

    def fused_path() -> None:
        minimax_m3_qknorm_rope_cache_shuffle_insert(
            qkv,
            weights[0],
            weights[1],
            cos_sin,
            positions,
            num_heads,
            num_kv_heads,
            num_index_heads,
            rotary_dim,
            eps,
            slot_mapping,
            fused_k,
            fused_v,
            fused_q,
            index_q_norm_weight=weights[2],
            index_k_norm_weight=weights[3],
            index_slot_mapping=index_slot_mapping,
            index_cache=fused_index,
            index_q_out=fused_index_q,
            kv_cache_dtype="fp8",
            k_scale=k_scale,
            v_scale=v_scale,
        )

    baseline_ms = elapsed_samples_ms(baseline_path)
    fused_ms = elapsed_samples_ms(fused_path)
    baseline_summary = summarize(baseline_ms)
    fused_summary = summarize(fused_ms)
    per_layer_delta = baseline_summary["p50_ms"] - fused_summary["p50_ms"]
    return {
        "num_tokens": num_tokens,
        "baseline_three_kernel_path": baseline_summary,
        "fused_one_kernel_path": fused_summary,
        "speedup_x_p50": baseline_summary["p50_ms"] / fused_summary["p50_ms"],
        "projected_57_layer_p50_savings_ms": per_layer_delta * SPARSE_LAYERS,
    }


def main() -> None:
    result = {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "constants": {
            "sparse_layers": SPARSE_LAYERS,
            "sparse_block_size": SPARSE_BLOCK_SIZE,
            "main_page_size": MAIN_PAGE_SIZE,
            "topk": TOPK,
            "index_head_dim": HEAD_DIM,
        },
        "indexer": [
            indexer_case(seq_len, query_len)
            for query_len in (1, 4)
            for seq_len in (8192, 131072, 262144, 393216)
        ],
        "cache_insert": [cache_insert_case(n) for n in (1, 4)],
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
