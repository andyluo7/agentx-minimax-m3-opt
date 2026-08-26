#!/usr/bin/env python3
"""Benchmark gfx950 sliding decode with forced 2-D and 3-D dispatch."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch

unified_attention_module = importlib.import_module(
    "aiter.ops.triton.attention.unified_attention"
)
from aiter.ops.triton.utils.types import e4m3_dtype

BLOCK_SIZE = 128
HEAD_SIZE = 128
NUM_QUERY_HEADS = 16
NUM_KV_HEADS = 1
WINDOW = 32768


def elapsed_ms(fn: Callable[[], None], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def source_sha256() -> dict[str, str]:
    wrapper = Path(unified_attention_module.__file__).resolve()
    kernel = (
        wrapper.parent.parent / "_triton_kernels" / "attention" / "unified_attention.py"
    )
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (wrapper, kernel)
    }


def run_case(
    batch_size: int, seq_len: int, warmup: int, repeats: int
) -> dict[str, float | int | bool]:
    torch.manual_seed(20260825 + batch_size * 1000003 + seq_len)
    device = torch.device("cuda")
    num_blocks = math.ceil(seq_len / BLOCK_SIZE)
    query = torch.randn(
        batch_size,
        NUM_QUERY_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_SIZE,
        device=device,
        dtype=torch.bfloat16,
    ).to(e4m3_dtype)
    value = torch.randn_like(key, dtype=torch.bfloat16).to(e4m3_dtype)
    block_table = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
        .expand(batch_size, -1)
        .contiguous()
    )
    cu_seqlens_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    seqused_k = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    scale = HEAD_SIZE**-0.5
    k_descale = torch.ones(1, device=device, dtype=torch.float32)
    v_descale = torch.ones(1, device=device, dtype=torch.float32)
    outputs = {mode: torch.empty_like(query) for mode in ("2d", "3d")}

    original_use_2d_kernel = unified_attention_module.use_2d_kernel

    def invoke(mode: str) -> None:
        unified_attention_module.use_2d_kernel = (
            (lambda *args, **kwargs: True)
            if mode == "2d"
            else (lambda *args, **kwargs: False)
        )
        try:
            unified_attention_module.unified_attention(
                q=query,
                k=key,
                v=value,
                out=outputs[mode],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=scale,
                causal=True,
                window_size=(WINDOW - 1, 0),
                block_table=block_table,
                softcap=0,
                q_descale=None,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        finally:
            unified_attention_module.use_2d_kernel = original_use_2d_kernel

    invoke("2d")
    invoke("3d")
    torch.cuda.synchronize()

    first = max(0, seq_len - WINDOW)
    live_key = key.view(-1, NUM_KV_HEADS, HEAD_SIZE)[first:seq_len].float()
    live_value = value.view(-1, NUM_KV_HEADS, HEAD_SIZE)[first:seq_len].float()
    live_key = live_key.repeat_interleave(NUM_QUERY_HEADS, dim=1)
    live_value = live_value.repeat_interleave(NUM_QUERY_HEADS, dim=1)
    scores = torch.einsum("bhd,khd->bhk", query.float() * scale, live_key)
    probs = torch.softmax(scores, dim=-1)
    reference = torch.einsum("bhk,khd->bhd", probs, live_value).to(torch.bfloat16)

    max_abs_2d_3d = (outputs["2d"].float() - outputs["3d"].float()).abs().max()
    max_abs_2d_ref = (outputs["2d"].float() - reference.float()).abs().max()
    max_abs_3d_ref = (outputs["3d"].float() - reference.float()).abs().max()
    torch.testing.assert_close(outputs["3d"], outputs["2d"], atol=1.5e-2, rtol=1e-2)
    torch.testing.assert_close(outputs["3d"], reference, atol=1.5e-1, rtol=1.5e-1)

    median_2d_ms = elapsed_ms(lambda: invoke("2d"), warmup, repeats)
    median_3d_ms = elapsed_ms(lambda: invoke("3d"), warmup, repeats)
    target_num_programs = unified_attention_module.get_num_sms() * 4
    auto_uses_2d = original_use_2d_kernel(
        HEAD_SIZE,
        WINDOW,
        True,
        1,
        seq_len,
        target_num_programs,
        batch_size * NUM_KV_HEADS,
    )
    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "window": WINDOW,
        "auto_uses_2d": bool(auto_uses_2d),
        "max_abs_2d_vs_3d": max_abs_2d_3d.item(),
        "max_abs_2d_vs_reference": max_abs_2d_ref.item(),
        "max_abs_3d_vs_reference": max_abs_3d_ref.item(),
        "median_2d_ms": median_2d_ms,
        "median_3d_ms": median_3d_ms,
        "speedup_3d_over_2d": median_2d_ms / median_3d_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--batch-sizes", default="1:8:32")
    parser.add_argument("--sequence-lengths", default="8192:32768:81920:262144")
    args = parser.parse_args()

    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(":"))
    sequence_lengths = tuple(int(value) for value in args.sequence_lengths.split(":"))
    cases = [
        run_case(batch_size, seq_len, args.warmup, args.repeats)
        for batch_size in batch_sizes
        for seq_len in sequence_lengths
    ]
    print(
        json.dumps(
            {
                "status": "PASS",
                "label": args.label,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "sources": source_sha256(),
                "cases": cases,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
