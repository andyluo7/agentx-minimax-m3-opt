#!/usr/bin/env python3
"""Verify MI355X AgentX aggregate artifacts against the fixed B200 gates."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Any


def nested(data: dict[str, Any], *keys: str) -> Any:
    value: Any = data
    for key in keys:
        value = value[key]
    return value


def profiled_error_count(aggregate_path: Path, aggregate_errors: int) -> tuple[int, str]:
    """Count errors in the measured phase when the raw export is available."""
    export = aggregate_path.parent / "aiperf_artifacts" / "profile_export.jsonl"
    if not export.is_file():
        return aggregate_errors, "aggregate_total"

    profiled_errors = 0
    try:
        with export.open() as lines:
            for line in lines:
                record = json.loads(line)
                phase = record.get("metadata", {}).get("benchmark_phase")
                if record.get("error") and phase != "warmup":
                    profiled_errors += 1
    except (OSError, json.JSONDecodeError):
        return aggregate_errors, "aggregate_total"
    return profiled_errors, "profile_export"


def post_profile_jit_events(aggregate_path: Path) -> tuple[int | None, str]:
    """Count serving-process JIT warnings after the measured phase starts."""
    aiperf_log = aggregate_path.parent / "aiperf_artifacts" / "logs" / "aiperf.log"
    server_log = aggregate_path.parent / "server.log"
    if not aiperf_log.is_file() or not server_log.is_file():
        return None, "profiling boundary or server log is missing"

    start: datetime | None = None
    try:
        for line in aiperf_log.read_text(errors="replace").splitlines():
            if "Phase profiling (profiling) started" not in line:
                continue
            match = re.search(r"\b(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if match:
                start = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                break
    except OSError:
        return None, "profiling boundary is unreadable"
    if start is None:
        return None, "profiling boundary timestamp is unproven"

    count = 0
    try:
        for line in server_log.read_text(errors="replace").splitlines():
            if "JIT compilation during inference" not in line:
                continue
            match = re.search(r"\b(\d{2}-\d{2} \d{2}:\d{2}:\d{2})\b", line)
            if not match:
                continue
            event = datetime.strptime(
                f"{start.year}-{match.group(1)}", "%Y-%m-%d %H:%M:%S"
            )
            if event >= start:
                count += 1
    except OSError:
        return None, "server log is unreadable"
    return count, "server_log"


def replay_exit_status(aggregate_path: Path) -> tuple[int | None, str]:
    """Read the replay command's exit status from the captured console log."""
    candidates = [aggregate_path.parent / "console.log"]
    candidates.extend(sorted(aggregate_path.parent.glob("slurm-*.out")))
    for path in candidates:
        if not path.is_file():
            continue
        try:
            matches = re.findall(
                r"(?m)^\+ replay_rc=(\d+)\s*$", path.read_text(errors="replace")
            )
        except OSError:
            continue
        if matches:
            return int(matches[-1]), path.name
    return None, "captured console log"


def canonical_agentx_status(
    aggregate_path: Path,
) -> tuple[bool, list[str], int | None, str, int | None, str]:
    """Require the checked-in canonical replay duration and warmup contract."""
    metadata: dict[str, str] = {}
    metadata_path = aggregate_path.parent / "run-metadata.txt"
    if metadata_path.is_file():
        for line in metadata_path.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator:
                metadata[key] = value

    command = ""
    command_path = aggregate_path.parent / "benchmark_command.txt"
    if command_path.is_file():
        command = command_path.read_text()

    duration: int | None = None
    if metadata.get("duration_seconds", "").isdigit():
        duration = int(metadata["duration_seconds"])
    if duration is None:
        match = re.search(r"--benchmark-duration(?:=|\s+)(\d+)", command)
        if match:
            duration = int(match.group(1))

    warmup: int | None = None
    if metadata.get("warmup_requests_per_lane", "").isdigit():
        warmup = int(metadata["warmup_requests_per_lane"])
    if warmup is None:
        match = re.search(r"--warmup-requests-per-lane(?:=|\s+)(\d+)", command)
        if match:
            warmup = int(match.group(1))

    reasons: list[str] = []
    if metadata.get("fast_mode") == "1":
        reasons.append("fast_mode=1")
    if duration is None:
        reasons.append("benchmark duration is unproven")
    elif duration < 3600:
        reasons.append(f"benchmark duration {duration}s is below canonical 3600s")
    if warmup is None:
        reasons.append("warmup requests per lane are unproven")
    elif warmup < 10:
        reasons.append(f"warmup requests per lane {warmup} is below canonical 10")
    jit_events, jit_source = post_profile_jit_events(aggregate_path)
    if metadata.get("jit_monitor_verbose") != "1":
        reasons.append("verbose serving-process JIT monitoring is unproven")
    if jit_events is None:
        reasons.append(jit_source)
    elif jit_events:
        reasons.append(f"{jit_events} post-profile JIT events")
    replay_rc, replay_rc_source = replay_exit_status(aggregate_path)
    if replay_rc is None:
        reasons.append("replay exit status is unproven")
    elif replay_rc:
        reasons.append(f"replay exited with status {replay_rc}")
    return (
        not reasons,
        reasons,
        jit_events,
        jit_source,
        replay_rc,
        replay_rc_source,
    )


def load_points(root: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
            throughput = nested(
                data, "request_metrics", "throughput", "per_gpu", "total_tput_tps"
            )
            normalized = nested(
                data, "request_metrics", "latency", "e2e_norm_intvty", "p90"
            )
            raw = nested(data, "request_metrics", "latency", "intvty", "p90")
            tpot_s = nested(data, "request_metrics", "latency", "tpot", "p90")
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        successful = int(data.get("num_requests_successful", 0))
        errors = int(data.get("request_accounting", {}).get("records_error_dropped", 0))
        profiled_errors, error_source = profiled_error_count(path, errors)
        (
            canonical,
            canonical_reasons,
            jit_events,
            jit_source,
            replay_rc,
            replay_rc_source,
        ) = canonical_agentx_status(path)
        hardware = str(data.get("hw", ""))
        target_point = (
            "mi355x" in hardware.lower()
            and str(data.get("framework", "")).lower() == "vllm"
            and str(data.get("precision", "")).lower() == "fp4"
            and str(data.get("infmax_model_prefix", "")).lower() == "minimaxm3"
        )
        points.append(
            {
                "path": str(path),
                "hw": hardware,
                "conc": int(data["conc"]),
                "throughput": float(throughput),
                "normalized": float(normalized),
                "raw": float(raw),
                "tpot_ms": float(tpot_s) * 1000.0,
                "successful": successful,
                "errors": errors,
                "profiled_errors": profiled_errors,
                "profiled_error_source": error_source,
                "canonical": canonical,
                "canonical_reasons": canonical_reasons,
                "post_profile_jit_events": jit_events,
                "post_profile_jit_source": jit_source,
                "replay_rc": replay_rc,
                "replay_rc_source": replay_rc_source,
                "target_point": target_point,
            }
        )
    return points


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("b200_reference.json"),
    )
    parser.add_argument(
        "--allow-diagnostic",
        action="store_true",
        help="Allow fast or sub-3600-second artifacts for diagnostic gate checks.",
    )
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())["primary_gates"]
    points = load_points(args.artifact_root)
    valid = [
        p
        for p in points
        if p["successful"] > 0
        and p["profiled_errors"] == 0
        and p["target_point"]
        and (args.allow_diagnostic or p["canonical"])
    ]
    c1 = [p for p in valid if p["conc"] == 1]
    c1_gate = reference["c1"]
    c1_pass = [
        p
        for p in c1
        if p["throughput"] >= c1_gate["mi355x_min_total_throughput_per_chip_tps"]
        and p["raw"] >= c1_gate["mi355x_min_p90_raw_interactivity_tps_per_user"]
        and p["normalized"]
        >= c1_gate["mi355x_min_p90_normalized_interactivity_tps_per_user"]
        and p["tpot_ms"] <= c1_gate["mi355x_max_p90_tpot_ms"]
    ]

    qos_gate = reference["comparable_qos"]
    qos_candidates = [
        p
        for p in valid
        if p["normalized"]
        >= qos_gate["mi355x_min_normalized_interactivity_tps_per_user"]
    ]
    qos_best = max(qos_candidates, key=lambda p: p["throughput"], default=None)
    qos_pass = bool(
        qos_best
        and qos_best["throughput"]
        >= qos_gate["mi355x_min_total_throughput_per_chip_tps"]
    )

    report = {
        "acceptance_mode": "diagnostic" if args.allow_diagnostic else "canonical",
        "valid_points": len(valid),
        "invalid_or_error_points": len(points) - len(valid),
        "noncanonical_points": sum(not p["canonical"] for p in points),
        "foreign_or_mismatched_points": sum(not p["target_point"] for p in points),
        "c1_gate_pass": bool(c1_pass),
        "best_c1": max(c1, key=lambda p: p["raw"], default=None),
        "comparable_qos_gate_pass": qos_pass,
        "best_comparable_qos": qos_best,
        "overall_pass": bool(c1_pass) and qos_pass,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
