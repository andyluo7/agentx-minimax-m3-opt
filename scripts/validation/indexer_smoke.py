#!/usr/bin/env python3
"""GPU correctness smoke for AITER PR 4787 decode score and top-k."""

from __future__ import annotations

import json
import os

import torch
from aiter.ops.sparse_attention import (
    pa_sparse_block_score_decode,
    pa_sparse_block_topk,
)


def run_case(dtype: torch.dtype) -> dict[str, object]:
    torch.manual_seed(7)
    device = torch.device("cuda")
    block_size = 128
    head_dim = 128
    query_len = 4
    num_blocks = 4
    seq_len = num_blocks * block_size

    q = (torch.randn(query_len, 1, head_dim, device=device) * 0.25).to(dtype)
    cache = (torch.randn(num_blocks, block_size, head_dim, device=device) * 0.25).to(
        dtype
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
    score = torch.full((1, query_len, 64), float("nan"), device=device)

    pa_sparse_block_score_decode(
        q,
        cache,
        score,
        block_table,
        seq_lens,
        query_len=query_len,
        max_seq_len=seq_len,
    )
    torch.cuda.synchronize()

    reference = torch.empty((query_len, num_blocks), device=device)
    prefix = seq_len - query_len
    q32 = q.float()[:, 0]
    k32 = cache.float()
    for row in range(query_len):
        visible = prefix + row + 1
        for block in range(num_blocks):
            live = max(0, min(block_size, visible - block * block_size))
            if live:
                reference[row, block] = torch.max(q32[row] @ k32[block, :live].T)
            else:
                reference[row, block] = float("-inf")

    observed = score[0, :, :num_blocks]
    max_abs_error = torch.max(torch.abs(observed - reference)).item()
    score_scale = torch.sum(observed * reference) / torch.sum(reference * reference)
    normalized = observed / score_scale
    max_abs_normalized_error = torch.max(torch.abs(normalized - reference)).item()
    if score_scale <= 0 or not torch.allclose(
        normalized, reference, atol=0.08, rtol=0.02
    ):
        raise AssertionError(
            f"score mismatch: scale={score_scale.item()}, "
            f"max_abs_normalized_error={max_abs_normalized_error}\n"
            f"observed={observed}\nreference={reference}"
        )

    topk = 2
    topk_idx = torch.empty((1, query_len, topk), device=device, dtype=torch.int32)
    sparse_bt = torch.empty((query_len, topk), device=device, dtype=torch.int32)
    sparse_ctx = torch.empty(query_len, device=device, dtype=torch.int32)
    pa_sparse_block_topk(
        score,
        topk_idx,
        block_table,
        seq_lens,
        sparse_bt,
        sparse_ctx,
        max_seq_len=seq_len,
        block_size=block_size,
        query_len=query_len,
        num_kv_heads=1,
        pages_per_block=1,
    )
    torch.cuda.synchronize()

    expected_topk = torch.topk(reference, topk, dim=1).indices
    for row in range(query_len):
        if set(topk_idx[0, row].tolist()) != set(expected_topk[row].tolist()):
            raise AssertionError(
                f"top-k mismatch at row {row}: "
                f"observed={topk_idx[0, row].tolist()} "
                f"expected={expected_topk[row].tolist()}"
            )

    return {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "query_len": query_len,
        "num_blocks": num_blocks,
        "max_abs_score_error": max_abs_error,
        "score_scale": score_scale.item(),
        "max_abs_normalized_score_error": max_abs_normalized_error,
    }


def main() -> None:
    # vLLM's MiniMax M3 FP8 index cache uses OCP E4M3FN. Test it first, then
    # exercise FNUZ as a compatibility guard for existing ROCm deployments.
    selector = os.getenv("INDEXER_SMOKE_DTYPE", "both")
    if selector not in {"both", "fn", "fnuz"}:
        raise ValueError(f"Unsupported INDEXER_SMOKE_DTYPE={selector}")
    dtypes = [torch.float8_e4m3fn] if selector in {"both", "fn"} else []
    if selector in {"both", "fnuz"} and hasattr(torch, "float8_e4m3fnuz"):
        dtypes.append(torch.float8_e4m3fnuz)

    print(json.dumps([run_case(dtype) for dtype in dtypes], sort_keys=True))


if __name__ == "__main__":
    main()
