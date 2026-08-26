#!/usr/bin/env python3
"""Correctness smoke for the fused MiniMax-M3 page-16 cache insert."""

from __future__ import annotations

import json

import torch
from aiter import reshape_and_cache
from aiter.ops.minimax_m3_fused_qknorm_rope import (
    minimax_m3_qknorm_rope_cache_shuffle_insert,
)
from vllm.models.minimax_m3.amd.ops.sparse_pa import (
    minimax_m3_insert_index_cache,
)

from vllm import _custom_ops as ops


def allocate_caches(
    num_pages: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_size = 16
    x = 16 // torch.empty((), dtype=dtype).element_size()
    device = torch.device("cuda")
    key = torch.zeros(
        num_pages,
        num_kv_heads,
        head_dim // x,
        page_size,
        x,
        device=device,
        dtype=dtype,
    )
    value = torch.zeros(
        num_pages,
        num_kv_heads,
        page_size // x,
        head_dim,
        x,
        device=device,
        dtype=dtype,
    )
    index = torch.zeros(
        num_pages,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    return key, value, index


def main() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache_dtype = torch.float8_e4m3fn
    num_tokens = 4  # target token plus three EAGLE3 speculative tokens
    num_heads = 16
    num_kv_heads = 1
    num_index_heads = 1
    head_dim = 128
    rotary_dim = 64
    eps = 1e-6
    num_pages = 32

    row_width = (num_heads + 2 * num_kv_heads + num_index_heads + 1) * head_dim
    qkv = torch.randn(num_tokens, row_width, device=device, dtype=dtype)
    weights = [
        torch.randn(head_dim, device=device, dtype=dtype) * 0.1 for _ in range(4)
    ]
    cos_sin = torch.randn(8192, rotary_dim, device=device, dtype=dtype)
    positions = torch.tensor([4096, 4097, 4098, 4099], device=device)
    slot_mapping = torch.tensor([15, 16, 31, 47], device=device)
    index_slot_mapping = torch.tensor([0, 17, 32, 63], device=device)
    k_scale = torch.tensor([1.0], device=device)
    v_scale = torch.tensor([1.0], device=device)

    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    index_q_size = num_index_heads * head_dim

    ref_qkv = qkv.clone()
    ref_q = torch.empty(num_tokens, q_size, device=device, dtype=dtype)
    ref_index_q = torch.empty(
        num_tokens, index_q_size, device=device, dtype=cache_dtype
    )
    ref_k, ref_v, ref_index = allocate_caches(
        num_pages, num_kv_heads, head_dim, cache_dtype
    )
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        ref_qkv,
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
        q_out=ref_q,
        index_q_out=ref_index_q,
        kv_cache_dtype="fp8",
    )
    k_start = q_size
    v_start = k_start + kv_size
    index_k_start = v_start + kv_size + index_q_size
    ref_key = ref_qkv[:, k_start:v_start].view(num_tokens, num_kv_heads, head_dim)
    ref_value = ref_qkv[:, v_start : v_start + kv_size].view(
        num_tokens, num_kv_heads, head_dim
    )
    ref_index_key = ref_qkv[:, index_k_start : index_k_start + head_dim]
    reshape_and_cache(
        ref_key.contiguous(),
        ref_value.contiguous(),
        ref_k,
        ref_v,
        slot_mapping,
        kv_cache_dtype="fp8",
        k_scale=k_scale,
        v_scale=v_scale,
        asm_layout=True,
    )
    minimax_m3_insert_index_cache(ref_index_key, ref_index, index_slot_mapping)

    got_q = torch.empty_like(ref_q)
    got_index_q = torch.empty_like(ref_index_q)
    got_k, got_v, got_index = allocate_caches(
        num_pages, num_kv_heads, head_dim, cache_dtype
    )
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
        got_k,
        got_v,
        got_q,
        index_q_norm_weight=weights[2],
        index_k_norm_weight=weights[3],
        index_slot_mapping=index_slot_mapping,
        index_cache=got_index,
        index_q_out=got_index_q,
        kv_cache_dtype="fp8",
        k_scale=k_scale,
        v_scale=v_scale,
    )
    torch.cuda.synchronize()

    tensors = {
        "q": (ref_q, got_q),
        "index_q": (ref_index_q, got_index_q),
        "key_cache": (ref_k, got_k),
        "value_cache": (ref_v, got_v),
        "index_cache": (ref_index, got_index),
    }
    errors: dict[str, float] = {}
    for name, (reference, observed) in tensors.items():
        max_abs = torch.max(torch.abs(reference.float() - observed.float())).item()
        errors[f"full_index.{name}"] = max_abs
        torch.testing.assert_close(
            observed.float(), reference.float(), rtol=8e-3, atol=8e-3
        )

    # Most sparse layers reuse the index selection from a preceding layer and
    # therefore take the skip-index branch. It has a distinct fused-kernel
    # argument layout, so validate it independently from the full-index case.
    skip_ref_qkv = qkv.clone()
    skip_ref_q = torch.empty_like(ref_q)
    skip_ref_k, skip_ref_v, _ = allocate_caches(
        num_pages, num_kv_heads, head_dim, cache_dtype
    )
    ops.fused_minimax_m3_qknorm_rope_kv_insert(
        skip_ref_qkv,
        weights[0],
        weights[1],
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        num_index_heads=num_index_heads,
        q_out=skip_ref_q,
        skip_index_branch=True,
    )
    skip_ref_key = skip_ref_qkv[:, k_start:v_start].view(
        num_tokens, num_kv_heads, head_dim
    )
    skip_ref_value = skip_ref_qkv[:, v_start : v_start + kv_size].view(
        num_tokens, num_kv_heads, head_dim
    )
    reshape_and_cache(
        skip_ref_key.contiguous(),
        skip_ref_value.contiguous(),
        skip_ref_k,
        skip_ref_v,
        slot_mapping,
        kv_cache_dtype="fp8",
        k_scale=k_scale,
        v_scale=v_scale,
        asm_layout=True,
    )

    skip_qkv = qkv.clone()
    skip_got_q = torch.empty_like(ref_q)
    skip_got_k, skip_got_v, _ = allocate_caches(
        num_pages, num_kv_heads, head_dim, cache_dtype
    )
    minimax_m3_qknorm_rope_cache_shuffle_insert(
        skip_qkv,
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
        skip_got_k,
        skip_got_v,
        skip_got_q,
        kv_cache_dtype="fp8",
        k_scale=k_scale,
        v_scale=v_scale,
        skip_index_branch=True,
    )
    torch.cuda.synchronize()

    skip_tensors = {
        "q": (skip_ref_q, skip_got_q),
        "key_cache": (skip_ref_k, skip_got_k),
        "value_cache": (skip_ref_v, skip_got_v),
    }
    for name, (reference, observed) in skip_tensors.items():
        max_abs = torch.max(torch.abs(reference.float() - observed.float())).item()
        errors[f"skip_index.{name}"] = max_abs
        torch.testing.assert_close(
            observed.float(), reference.float(), rtol=8e-3, atol=8e-3
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "device": torch.cuda.get_device_name(0),
                "cache_dtype": str(cache_dtype),
                "num_tokens": num_tokens,
                "cases": ["full_index", "skip_index"],
                "max_abs_errors": errors,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
