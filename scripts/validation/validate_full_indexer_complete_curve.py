#!/usr/bin/env python3
"""Validate the eight-point MiniMax-M3 full-indexer MI355X AgentX curve."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from verify_70pct import canonical_agentx_status, profiled_error_count


EXPECTED_POINTS = {
    "c1_tp4_resident": (1, "none", "none"),
    "c5_tp4_resident": (5, "none", "none"),
    "c10_tp4_resident": (10, "none", "none"),
    "c15_tp4_vllm_simple": (15, "dram", "vllm-simple"),
    "c20_tp4_vllm_simple": (20, "dram", "vllm-simple"),
    "c25_tp4_vllm_simple": (25, "dram", "vllm-simple"),
    "c30_tp4_vllm_simple": (30, "dram", "vllm-simple"),
    "c32_tp4_vllm_simple": (32, "dram", "vllm-simple"),
}

REQUIRED_METADATA = {
    "recipe_sha256": "76f025a44df07ff54ea4ceb6ec076f40403fa1a091e0ff1c03d4aa63b76a670d",
    "container_registry_digest": "sha256:bb44b39aea26798cce43030a98bf48efd0322ca7147367db86e38b96bd80f0e7",
    "model_revision": "b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a",
    "draft_model_revision": "96692486b5fd38ebf8fd2a5f6bb53427d30819a8",
    "max_num_seqs": "256",
    "max_num_batched_tokens": "32768",
    "max_cudagraph_capture_size": "512",
    "num_speculative_tokens": "4",
    "synthetic_acceptance_length": "3.02",
    "target_attention_backend": "ROCM_AITER_UNIFIED_ATTN",
    "draft_attention_backend": "ROCM_AITER_UNIFIED_ATTN",
    "draft_attention_window": "32768",
    "aiter_sliding_decode_3d": "1",
    "indexer_kv_dtype": "fp8",
    "target_kv_cache_dtype": "fp8",
    "aiter_unified_attn_kernel": "aiter",
    "aiter_unified_attn_cache_writer": "aiter",
    "aiter_fused_cache_insert": "1",
    "aiter_fused_ar_gemma": "1",
    "agentx_jit_warmup": "1",
    "aiter_sparse_precompile": "1",
}

ALLOWED_WARMUP_SHA256 = {
    # v19 direct synthetic-shape warmup used by the original sweep.
    "cc39c7f24cf8246aa1cbac5ec0b9b640f1573379641d94373c8365400aea3c44",
    # v23 adds TP4 production-buffer attention warmup after C15 job 1406
    # exposed a late one-sequence/eight-segment reduction specialization.
    "14be2442af1cde419dff81f0a7d851f1ba34831a557dc39ac69658a4f8b81037",
}

FAULT_PATTERNS = (
    re.compile(r"hipErrorIllegalAddress", re.IGNORECASE),
    re.compile(r"illegal memory access", re.IGNORECASE),
    re.compile(r"HSA_STATUS_ERROR", re.IGNORECASE),
    re.compile(r"memory access fault", re.IGNORECASE),
    re.compile(r"GPU fault detected", re.IGNORECASE),
    re.compile(r"segmentation fault", re.IGNORECASE),
    re.compile(r"EngineCoreDeadError", re.IGNORECASE),
)


def normalize_backend(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name")
    return str(value or "none")


def read_metadata(directory: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    path = directory / "run-metadata.txt"
    if not path.is_file():
        return metadata
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            metadata[key] = value
    return metadata


def aggregate_candidates(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    candidates = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
            data["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]
            data["request_metrics"]["latency"]["e2e_norm_intvty"]["p90"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        candidates.append((path, data))
    return candidates


def gpu_fault_events(directory: Path) -> tuple[int | None, str]:
    paths = [directory / "server.log", directory / "console.log"]
    readable = [path for path in paths if path.is_file()]
    if not readable:
        return None, "server/console logs are missing"
    matches: set[tuple[str, int, str]] = set()
    for path in readable:
        for line_number, line in enumerate(
            path.read_text(errors="replace").splitlines(), start=1
        ):
            if any(pattern.search(line) for pattern in FAULT_PATTERNS):
                matches.add((path.name, line_number, line.strip()))
    return len(matches), ",".join(path.name for path in readable)


def validate_point(path: Path, data: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    directory = path.parent
    metadata = read_metadata(directory)
    command_path = directory / "vllm_command.txt"
    command = command_path.read_text(errors="replace") if command_path.is_file() else ""
    conc = int(data.get("conc", 0))
    tp = int(data.get("tp") or metadata.get("tp") or 0)
    offload = str(data.get("kv_offloading") or metadata.get("kv_offloading") or "none")
    backend = normalize_backend(
        data.get("kv_offload_backend") or metadata.get("kv_offload_backend")
    )
    key = next(
        (
            name
            for name, expected in EXPECTED_POINTS.items()
            if (conc, offload, backend) == expected
        ),
        None,
    )

    aggregate_errors = int(data.get("request_accounting", {}).get("records_error_dropped", 0))
    measured_errors, error_source = profiled_error_count(path, aggregate_errors)
    canonical, canonical_reasons, jit_events, jit_source, replay_rc, replay_source = (
        canonical_agentx_status(path)
    )
    faults, fault_source = gpu_fault_events(directory)
    reasons: list[str] = []

    if key is None:
        reasons.append("point does not match the requested concurrency/offload tuple")
    if tp != 4:
        reasons.append(f"tp={tp}, expected 4")
    if str(data.get("framework", "")).lower() != "vllm":
        reasons.append(f"framework={data.get('framework')!r}, expected vllm")
    if str(data.get("precision", "")).lower() != "fp4":
        reasons.append(f"precision={data.get('precision')!r}, expected fp4")
    if str(data.get("infmax_model_prefix", "")).lower() != "minimaxm3":
        reasons.append("model is not minimaxm3")
    for name, expected in REQUIRED_METADATA.items():
        if metadata.get(name) != expected:
            reasons.append(f"{name}={metadata.get(name)!r}, expected {expected!r}")
    warmup_sha256 = metadata.get("agentx_jit_warmup_sha256")
    if warmup_sha256 not in ALLOWED_WARMUP_SHA256:
        reasons.append(
            f"agentx_jit_warmup_sha256={warmup_sha256!r} is not an approved "
            "full-indexer warmup"
        )
    if not command:
        reasons.append("effective vLLM command is missing")
    if any(marker in command for marker in ("--hf-overrides", "index_topk_freq", "use_index_cache")):
        reasons.append("index-cache override is present; this is not a full-indexer run")
    expected_offload_bytes = 1499 * 1024**3
    if offload == "dram":
        if "SimpleCPUOffloadConnector" not in command:
            reasons.append("SimpleCPUOffloadConnector is missing")
        if f'"cpu_bytes_to_use":{expected_offload_bytes}' not in command.replace("\\", ""):
            reasons.append("offload allocation is not the expected 1499 GiB")
    elif "--kv-transfer-config" in command:
        reasons.append("resident point unexpectedly configures KV transfer")
    successful = int(data.get("num_requests_successful", 0))
    if successful <= 0:
        reasons.append("no successful profiling requests")
    if measured_errors:
        reasons.append(f"{measured_errors} measured request errors from {error_source}")
    if not canonical:
        reasons.extend(canonical_reasons)
    if faults is None:
        reasons.append(fault_source)
    elif faults:
        reasons.append(f"{faults} GPU/runtime fault events in {fault_source}")

    return key, {
        "pass": not reasons,
        "point": key,
        "conc": conc,
        "tp": tp,
        "kv_offloading": offload,
        "kv_offload_backend": backend,
        "successful_requests": successful,
        "aggregate_errors_including_warmup": aggregate_errors,
        "measured_errors": measured_errors,
        "measured_error_source": error_source,
        "post_profile_jit_events": jit_events,
        "post_profile_jit_source": jit_source,
        "gpu_fault_events": faults,
        "gpu_fault_source": fault_source,
        "replay_rc": replay_rc,
        "replay_rc_source": replay_source,
        "throughput_tok_s_chip": float(
            data["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]
        ),
        "p90_e2e_normalized_interactivity_tok_s_user": float(
            data["request_metrics"]["latency"]["e2e_norm_intvty"]["p90"]
        ),
        "p90_raw_interactivity_tok_s_user": float(
            data["request_metrics"]["latency"]["intvty"]["p90"]
        ),
        "p90_tpot_ms": float(data["request_metrics"]["latency"]["tpot"]["p90"])
        * 1000.0,
        "reasons": reasons,
        "artifact": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidates: dict[str, list[dict[str, Any]]] = {
        key: [] for key in EXPECTED_POINTS
    }
    unexpected: list[dict[str, Any]] = []
    for path, data in aggregate_candidates(args.artifact_root):
        key, validated = validate_point(path, data)
        if key is None:
            unexpected.append(validated)
        else:
            candidates[key].append(validated)

    points: dict[str, Any] = {}
    for key, validated in candidates.items():
        passing = [point for point in validated if point["pass"]]
        points[key] = (
            max(
                passing,
                key=lambda point: (
                    point["successful_requests"],
                    point["throughput_tok_s_chip"],
                ),
            )
            if passing
            else {
                "pass": False,
                "reasons": ["no passing aggregate"] if not validated else [],
                "candidates": validated,
            }
        )

    report = {
        "overall_pass": all(points[key]["pass"] for key in EXPECTED_POINTS),
        "full_indexer_required": True,
        "offload_allocation_gib": 1499,
        "expected_points": EXPECTED_POINTS,
        "performance_points": points,
        "unexpected_aggregates": unexpected,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
