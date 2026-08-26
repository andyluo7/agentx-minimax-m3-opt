#!/usr/bin/env python3
"""Validate gfx950 sliding-window 3-D dispatch against a Torch reference."""

from __future__ import annotations

import importlib
import json
import math

import torch

unified_attention_module = importlib.import_module(
    "aiter.ops.triton.attention.unified_attention"
)
from aiter.ops.triton.utils.types import e4m3_dtype

BLOCK_SIZE = 128
HEAD_SIZE = 128
NUM_QUERY_HEADS = 16
NUM_KV_HEADS = 1
SEQ_LEN = 8203
WINDOW = 2053


def shuffle_kv_cache(
    key_cache: torch.Tensor, value_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    k_width = 16 // key_cache.element_size()
    key_cache = key_cache.permute(0, 2, 3, 1).reshape(
        -1, NUM_KV_HEADS, HEAD_SIZE // k_width, k_width, BLOCK_SIZE
    )
    key_cache = key_cache.permute(0, 1, 2, 4, 3).contiguous()
    value_cache = value_cache.permute(0, 2, 1, 3).reshape(
        -1, NUM_KV_HEADS, BLOCK_SIZE // k_width, k_width, HEAD_SIZE
    )
    value_cache = value_cache.permute(0, 1, 2, 4, 3).contiguous()
    return key_cache, value_cache


def run_case(q_dtype: torch.dtype, shuffled: bool) -> dict[str, object]:
    print(
        json.dumps(
            {
                "event": "start_case",
                "q_dtype": str(q_dtype),
                "shuffled_kv_cache": shuffled,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    torch.manual_seed(20260825 + 1000003 + SEQ_LEN + int(shuffled))
    print("seeded", flush=True)
    device = torch.device("cuda")
    num_blocks = math.ceil(SEQ_LEN / BLOCK_SIZE)
    query = torch.randn(
        1, NUM_QUERY_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16
    ).to(q_dtype)
    print("query_ready", flush=True)
    key_orig = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=torch.bfloat16,
    ).to(e4m3_dtype)
    value_orig = torch.randn_like(key_orig, dtype=torch.bfloat16).to(e4m3_dtype)
    key, value = (
        shuffle_kv_cache(key_orig, value_orig) if shuffled else (key_orig, value_orig)
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([SEQ_LEN], device=device, dtype=torch.int32)
    output = torch.empty(
        1, NUM_QUERY_HEADS, HEAD_SIZE, device=device, dtype=torch.bfloat16
    )
    q_descale = (
        torch.ones(1, device=device, dtype=torch.float32)
        if q_dtype == e4m3_dtype
        else None
    )
    k_descale = torch.ones(1, device=device, dtype=torch.float32)
    v_descale = torch.ones(1, device=device, dtype=torch.float32)

    auto_uses_2d = unified_attention_module.use_2d_kernel(
        HEAD_SIZE,
        WINDOW,
        True,
        1,
        SEQ_LEN,
        unified_attention_module.get_num_sms() * 4,
        NUM_KV_HEADS,
    )
    assert not auto_uses_2d
    print("inputs_ready", flush=True)
    unified_attention_module.unified_attention(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        seqused_k=seqused_k,
        max_seqlen_k=SEQ_LEN,
        softmax_scale=HEAD_SIZE**-0.5,
        causal=True,
        window_size=(WINDOW - 1, 0),
        block_table=block_table,
        softcap=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        shuffled_kv_cache=shuffled,
    )
    torch.cuda.synchronize()
    print("attention_complete", flush=True)

    first = SEQ_LEN - WINDOW
    live_key = key_orig.view(-1, NUM_KV_HEADS, HEAD_SIZE)[first:SEQ_LEN]
    live_value = value_orig.view(-1, NUM_KV_HEADS, HEAD_SIZE)[first:SEQ_LEN]
    live_key = live_key.float().repeat_interleave(NUM_QUERY_HEADS, dim=1)
    live_value = live_value.float().repeat_interleave(NUM_QUERY_HEADS, dim=1)
    scores = torch.einsum("bhd,khd->bhk", query.float() * (HEAD_SIZE**-0.5), live_key)
    reference = torch.einsum(
        "bhk,khd->bhd", torch.softmax(scores, dim=-1), live_value
    ).to(torch.bfloat16)
    max_abs = (output.float() - reference.float()).abs().max().item()
    torch.testing.assert_close(
        output.float(), reference.float(), atol=1.5e-1, rtol=1.5e-1
    )
    return {
        "q_dtype": str(q_dtype),
        "shuffled_kv_cache": shuffled,
        "auto_uses_2d": bool(auto_uses_2d),
        "max_abs_vs_reference": max_abs,
    }


def main() -> None:
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    cases = [
        run_case(q_dtype, shuffled)
        for q_dtype in (torch.bfloat16, e4m3_dtype)
        for shuffled in (False, True)
    ]
    print(json.dumps({"status": "PASS", "cases": cases}, sort_keys=True))


if __name__ == "__main__":
    main()
