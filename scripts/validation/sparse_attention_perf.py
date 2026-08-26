#!/usr/bin/env python3
"""Compare the production Gluon sparse attend with AITER ASM alternatives."""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Callable

import torch
from aiter import dtypes as aiter_dtypes
from aiter import pa_fwd_asm
from vllm.models.minimax_m3.amd.ops.sparse_pa import _run_gluon_decode

HEAD_DIM = 128
NUM_Q_HEADS = 16
NUM_KV_HEADS = 1
PAGE_SIZE = 16
SELECTED_TOKENS = 16 * 128
SELECTED_PAGES = SELECTED_TOKENS // PAGE_SIZE
SPARSE_LAYERS = 57


def elapsed_samples_ms(
    fn: Callable[[], None], *, warmup: int = 8, repeats: int = 80
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


def allocate_fp8_caches(num_pages: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Match the dtype selected by AITER on the active ROCm platform. gfx950
    # uses the FNUZ encoding; treating the same bytes as OCP E4M3 invalidates
    # an ASM-vs-Gluon correctness comparison.
    dtype = aiter_dtypes.fp8
    x = 16 // torch.empty((), dtype=dtype).element_size()
    device = torch.device("cuda")
    key = (
        torch.randn(num_pages, NUM_KV_HEADS, HEAD_DIM // x, PAGE_SIZE, x, device=device)
        * 0.1
    ).to(dtype)
    value = (
        torch.randn(num_pages, NUM_KV_HEADS, PAGE_SIZE // x, HEAD_DIM, x, device=device)
        * 0.1
    ).to(dtype)
    return key, value


def run_case(
    num_tokens: int, scale_mode: str, high_precision: int
) -> dict[str, object]:
    device = torch.device("cuda")
    torch.manual_seed(41 + num_tokens)
    query = (torch.randn(num_tokens, NUM_Q_HEADS, HEAD_DIM, device=device) * 0.1).to(
        torch.bfloat16
    )
    key_cache, value_cache = allocate_fp8_caches(SELECTED_PAGES)
    block_table = torch.arange(SELECTED_PAGES, device=device, dtype=torch.int32)[
        None
    ].repeat(num_tokens, 1)
    context_lens = torch.full(
        (num_tokens,), SELECTED_TOKENS, device=device, dtype=torch.int32
    )
    k_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    v_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    per_token_scale = torch.ones(
        SELECTED_PAGES,
        NUM_KV_HEADS,
        PAGE_SIZE,
        device=device,
        dtype=torch.float32,
    )
    gluon_output = torch.empty_like(query)

    def gluon_path() -> None:
        _run_gluon_decode(
            query,
            key_cache,
            value_cache,
            block_table,
            context_lens,
            NUM_KV_HEADS,
            HEAD_DIM**-0.5,
            gluon_output,
            k_scale,
            v_scale,
        )

    gluon_samples = elapsed_samples_ms(gluon_path)
    result: dict[str, object] = {
        "num_tokens": num_tokens,
        "gluon_fp8": summarize(gluon_samples),
    }

    if scale_mode == "scalar":
        asm_k_scale = k_scale
        asm_v_scale = v_scale
    elif scale_mode == "per_token":
        asm_k_scale = per_token_scale
        asm_v_scale = per_token_scale
    else:
        raise ValueError(
            "ASM_SCALE_MODE must be scalar or per_token; null scales cause "
            "the packaged per-token FP8 kernel to access address zero"
        )

    asm_output = torch.empty_like(query)

    def asm_path() -> None:
        pa_fwd_asm(
            query,
            key_cache,
            value_cache,
            block_table,
            context_lens,
            block_table.stride(0),
            max_qlen=1,
            K_QScale=asm_k_scale,
            V_QScale=asm_v_scale,
            out_=asm_output,
            high_precision=high_precision,
        )

    asm_samples = elapsed_samples_ms(asm_path)
    error = torch.max(torch.abs(asm_output.float() - gluon_output.float())).item()
    gluon_norm = torch.linalg.vector_norm(gluon_output.float()).item()
    diff_norm = torch.linalg.vector_norm(
        asm_output.float() - gluon_output.float()
    ).item()
    asm_summary = summarize(asm_samples)
    result["asm"] = {
        "status": "PASS",
        "scale_mode": scale_mode,
        "high_precision": high_precision,
        "timing": asm_summary,
        "max_abs_error_vs_gluon": error,
        "relative_l2_error_vs_gluon": diff_norm / max(gluon_norm, 1e-12),
        "speedup_x_p50_vs_gluon": result["gluon_fp8"]["p50_ms"] / asm_summary["p50_ms"],  # type: ignore[index]
        "projected_57_layer_p50_delta_ms": (
            result["gluon_fp8"]["p50_ms"] - asm_summary["p50_ms"]  # type: ignore[index]
        )
        * SPARSE_LAYERS,
    }

    return result


def main() -> None:
    scale_mode = os.environ.get("ASM_SCALE_MODE", "per_token")
    high_precision = int(os.environ.get("ASM_HIGH_PRECISION", "1"))
    if high_precision not in (0, 1):
        raise ValueError("ASM_HIGH_PRECISION must be 0 or 1")
    print(
        json.dumps(
            {
                "status": "PASS",
                "device": torch.cuda.get_device_name(0),
                "selected_tokens": SELECTED_TOKENS,
                "scale_mode": scale_mode,
                "high_precision": high_precision,
                "cases": [
                    run_case(num_tokens, scale_mode, high_precision)
                    for num_tokens in (1, 4)
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
