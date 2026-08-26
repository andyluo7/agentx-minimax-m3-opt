#!/usr/bin/env python3
"""Warm a long MiniMax-M3 prefix, then capture a short decode profile."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


def post_json(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = b"" if payload is None else json.dumps(payload).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=1800) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def timed_request(url: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    response = post_json(url, payload)
    return response, time.perf_counter() - start


def main() -> None:
    port = int(os.environ.get("PORT", "8886"))
    base_url = f"http://127.0.0.1:{port}"
    target_prompt_tokens = int(os.environ.get("PROFILE_PROMPT_TOKENS", "262144"))
    warm_prompt_tokens = int(
        os.environ.get("PROFILE_WARM_PROMPT_TOKENS", str(target_prompt_tokens))
    )
    profile_max_tokens = int(os.environ.get("PROFILE_MAX_TOKENS", "128"))
    result_dir = Path(os.environ["RESULT_DIR"])
    result_dir.mkdir(parents=True, exist_ok=True)

    # The GPT-style tokenizer maps the repeated short word close to one token
    # per repetition. The observed API usage below is authoritative.
    prompt = "x " * target_prompt_tokens
    warm_prompt = "x " * warm_prompt_tokens
    common = {
        "model": "amd/MiniMax-M3-MXFP4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "stream": False,
        "ignore_eos": True,
    }

    warm_response, warm_seconds = timed_request(
        f"{base_url}/v1/chat/completions",
        {
            **common,
            "messages": [{"role": "user", "content": warm_prompt}],
            "max_tokens": 8,
        },
    )
    warm_usage = warm_response.get("usage", {})
    observed_prompt_tokens = int(warm_usage.get("prompt_tokens", 0))
    if observed_prompt_tokens < warm_prompt_tokens // 2:
        raise RuntimeError(
            "Synthetic prompt tokenization was unexpectedly short: "
            f"{observed_prompt_tokens} < {warm_prompt_tokens // 2}"
        )

    post_json(f"{base_url}/start_profile")
    try:
        profile_response, profile_seconds = timed_request(
            f"{base_url}/v1/chat/completions",
            {**common, "max_tokens": profile_max_tokens},
        )
    finally:
        post_json(f"{base_url}/stop_profile")

    summary = {
        "status": "PASS",
        "target_prompt_tokens": target_prompt_tokens,
        "warm_prompt_tokens": warm_prompt_tokens,
        "observed_warm_prompt_tokens": observed_prompt_tokens,
        "warm_completion_tokens": warm_usage.get("completion_tokens"),
        "warm_seconds": warm_seconds,
        "profile_usage": profile_response.get("usage", {}),
        "profile_seconds": profile_seconds,
    }
    (result_dir / "profile-request-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
