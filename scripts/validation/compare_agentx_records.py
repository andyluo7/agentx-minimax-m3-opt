#!/usr/bin/env python3
"""Compare identical successful AgentX requests in two AIPerf exports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def metric(record: dict[str, Any], name: str) -> float:
    return float(record["metrics"][name]["value"])


def load(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    records: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open() as lines:
        for line in lines:
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase") == "warmup":
                continue
            metrics = record.get("metrics", {})
            required = {
                "request_latency",
                "time_to_first_token",
                "full_response_inter_token_latency",
                "full_response_output_token_throughput_per_user",
            }
            if not required <= metrics.keys():
                continue
            trace_id = metadata.get("source_trace_id") or metadata.get(
                "conversation_id"
            )
            turn_index = metadata.get("turn_index")
            if trace_id is None or turn_index is None:
                continue
            records[(str(trace_id), int(turn_index))] = record
    return records


def median_ratio(
    candidate: dict[tuple[str, int], dict[str, Any]],
    reference: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
    metric_name: str,
) -> float:
    return statistics.median(
        metric(candidate[key], metric_name) / metric(reference[key], metric_name)
        for key in keys
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--reference-label", default="reference")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    candidate = load(args.candidate)
    reference = load(args.reference)
    common = sorted(candidate.keys() & reference.keys())
    if not common:
        raise SystemExit("no identical successful profiling requests found")

    report = {
        "candidate": args.candidate_label,
        "reference": args.reference_label,
        "candidate_records": len(candidate),
        "reference_records": len(reference),
        "exact_matches": len(common),
        "median_candidate_over_reference": {
            "ttft": median_ratio(
                candidate, reference, common, "time_to_first_token"
            ),
            "tpot": median_ratio(
                candidate,
                reference,
                common,
                "full_response_inter_token_latency",
            ),
            "e2e": median_ratio(candidate, reference, common, "request_latency"),
            "raw_interactivity": median_ratio(
                candidate,
                reference,
                common,
                "full_response_output_token_throughput_per_user",
            ),
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out:
        args.json_out.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
