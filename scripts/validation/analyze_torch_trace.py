"""Summarize accelerator kernels from one or more PyTorch Chrome traces."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FAMILY_PATTERNS = (
    (
        "moe",
        re.compile(
            r"moe|expert|sorting|grouped_topk|topk_softmax|moe1|moe2", re.IGNORECASE
        ),
    ),
    (
        "collective",
        re.compile(
            r"all.?reduce|reduce.?scatter|all.?gather|broadcast|nccl|rccl|"
            r"quick.?reduce|cross_device_reduce",
            re.IGNORECASE,
        ),
    ),
    (
        "indexer",
        re.compile(r"indexer|mqa_logits|sparse.*score|fp8_mqa|topk", re.IGNORECASE),
    ),
    (
        "sparse_attention",
        re.compile(
            r"sparse.*attn|sparse.*attention|paged.*attn|paged.*attention|"
            r"pa_(?:bf16|fp16).*pertoken|pa_decode_ps_reduce",
            re.IGNORECASE,
        ),
    ),
    (
        "dense_gemm",
        re.compile(
            r"gemm(?!a)|gemv|matmul|mm_kernel|bf16gemm|wvsplitk", re.IGNORECASE
        ),
    ),
    (
        "norm_rope_cache",
        re.compile(
            r"rms|norm|rope|rotary|cache.*insert|cache.*shuffle|reshape_and_cache",
            re.IGNORECASE,
        ),
    ),
    ("attention", re.compile(r"attention|attn|flash", re.IGNORECASE)),
    (
        "copy_cast",
        re.compile(r"direct_copy|bfloat16tofloat32|copy_kernel", re.IGNORECASE),
    ),
    (
        "quantization",
        re.compile(r"quant|dequant|(?<!no)cast|convert", re.IGNORECASE),
    ),
    ("memory", re.compile(r"memcpy|memset", re.IGNORECASE)),
)


def open_trace(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def trace_events(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    else:
        events = payload
    return (event for event in events if isinstance(event, dict))


def is_accelerator_kernel(event: dict[str, Any]) -> bool:
    if event.get("ph") != "X" or float(event.get("dur", 0)) <= 0:
        return False
    category = str(event.get("cat", "")).lower()
    return category in {"kernel", "gpu_memcpy", "gpu_memset"}


def family(name: str) -> str:
    for label, pattern in FAMILY_PATTERNS:
        if pattern.search(name):
            return label
    return "other"


def discover(inputs: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for input_path in inputs:
        if input_path.is_dir():
            found.update(input_path.rglob("*.json"))
            found.update(input_path.rglob("*.json.gz"))
            found.update(input_path.rglob("*.pt.trace.json"))
            found.update(input_path.rglob("*.pt.trace.json.gz"))
        else:
            found.add(input_path)
    return sorted(path for path in found if path.is_file())


def summarize(paths: list[Path]) -> dict[str, Any]:
    per_trace: list[dict[str, Any]] = []
    for path in paths:
        kernel_us: dict[str, float] = defaultdict(float)
        family_us: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for event in trace_events(open_trace(path)):
            if not is_accelerator_kernel(event):
                continue
            name = str(event.get("name", "<unnamed>"))
            duration_us = float(event["dur"])
            kernel_us[name] += duration_us
            family_us[family(name)] += duration_us
            counts[name] += 1
        per_trace.append(
            {
                "path": str(path),
                "total_kernel_us": sum(kernel_us.values()),
                "kernel_us": dict(kernel_us),
                "family_us": dict(family_us),
                "counts": dict(counts),
            }
        )

    divisor = max(1, len(per_trace))
    avg_kernel_us: dict[str, float] = defaultdict(float)
    avg_family_us: dict[str, float] = defaultdict(float)
    avg_counts: dict[str, float] = defaultdict(float)
    for trace in per_trace:
        for name, duration_us in trace["kernel_us"].items():
            avg_kernel_us[name] += duration_us / divisor
        for name, duration_us in trace["family_us"].items():
            avg_family_us[name] += duration_us / divisor
        for name, count in trace["counts"].items():
            avg_counts[name] += count / divisor

    return {
        "trace_count": len(per_trace),
        "paths": [trace["path"] for trace in per_trace],
        "mean_total_kernel_ms_per_trace": (
            sum(trace["total_kernel_us"] for trace in per_trace) / divisor / 1000
        ),
        "max_total_kernel_ms": max(
            (trace["total_kernel_us"] / 1000 for trace in per_trace), default=0
        ),
        "family_ms": {
            name: duration_us / 1000
            for name, duration_us in sorted(
                avg_family_us.items(), key=lambda item: item[1], reverse=True
            )
        },
        "kernels": [
            {
                "name": name,
                "mean_ms_per_trace": duration_us / 1000,
                "mean_calls_per_trace": avg_counts[name],
                "mean_us_per_call": duration_us / max(1.0, avg_counts[name]),
                "family": family(name),
            }
            for name, duration_us in sorted(
                avg_kernel_us.items(), key=lambda item: item[1], reverse=True
            )
        ],
    }


def print_summary(summary: dict[str, Any], top: int) -> None:
    total = float(summary["mean_total_kernel_ms_per_trace"])
    print(
        f"traces={summary['trace_count']} "
        f"mean_kernel_ms={total:.3f} "
        f"max_kernel_ms={summary['max_total_kernel_ms']:.3f}"
    )
    print("\nFamily | mean GPU ms/trace | share")
    print("--- | ---: | ---:")
    for name, duration_ms in summary["family_ms"].items():
        share = 100 * duration_ms / total if total else 0
        print(f"{name} | {duration_ms:.3f} | {share:.1f}%")

    print("\nKernel | family | mean ms/trace | calls/trace | us/call")
    print("--- | --- | ---: | ---: | ---:")
    for item in summary["kernels"][:top]:
        name = item["name"].replace("|", "\\|")
        print(
            f"{name} | {item['family']} | {item['mean_ms_per_trace']:.3f} | "
            f"{item['mean_calls_per_trace']:.1f} | {item['mean_us_per_call']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    paths = discover(args.trace)
    if not paths:
        raise SystemExit("No JSON or JSON.GZ traces found")
    summary = summarize(paths)
    print_summary(summary, args.top)
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
