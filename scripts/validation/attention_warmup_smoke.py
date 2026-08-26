#!/usr/bin/env python3
"""Compile the MiniMax-M3 AgentX AITER-attention warmup matrix twice."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import torch

from vllm.model_executor.warmup.minimax_m3_msa_warmup import (
    _warmup_aiter_unified_attention,
)
from vllm.v1.spec_decode.utils import eagle_prepare_next_token_padded_kernel


def warmup_eagle_next_token(device: torch.device) -> None:
    """Exercise bypass plus two-, three-, and four-draft EAGLE rows."""
    for num_reqs in (1, 2, 4, 8, 16, 32):
        for num_sampled_tokens in (1, 3, 4, 5):
            sampled = torch.zeros(
                (num_reqs, num_sampled_tokens),
                dtype=torch.int32,
                device=device,
            )
            discard = torch.zeros(num_reqs, dtype=torch.bool, device=device)
            backup = torch.zeros(num_reqs, dtype=torch.int32, device=device)
            next_tokens = torch.empty_like(backup)
            valid = torch.empty_like(backup)
            eagle_prepare_next_token_padded_kernel[(num_reqs,)](
                sampled,
                discard,
                backup,
                next_tokens,
                valid,
                200_064,
                num_sampled_tokens,
                num_reqs,
                sampled.stride(0),
                BLOCK_SIZE_TOKENS=1 << (num_sampled_tokens - 1).bit_length(),
            )
            torch.testing.assert_close(next_tokens, torch.zeros_like(next_tokens))
            torch.testing.assert_close(
                valid,
                torch.full_like(valid, num_sampled_tokens),
            )


def main() -> None:
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    text_config = SimpleNamespace(
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
    )
    worker = SimpleNamespace(
        device=torch.device("cuda:0"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=text_config),
            max_model_len=1_048_576,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
        cache_config=SimpleNamespace(block_size=128),
    )

    elapsed: list[float] = []
    for _ in range(2):
        start = time.monotonic()
        _warmup_aiter_unified_attention(worker)
        warmup_eagle_next_token(worker.device)
        torch.cuda.synchronize(worker.device)
        elapsed.append(time.monotonic() - start)

    print(
        json.dumps(
            {
                "status": "PASS",
                "device": torch.cuda.get_device_name(0),
                "first_pass_seconds": elapsed[0],
                "cached_pass_seconds": elapsed[1],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
