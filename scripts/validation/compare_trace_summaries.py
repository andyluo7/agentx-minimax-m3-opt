"""Compare two JSON summaries emitted by analyze_torch_trace.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pct_change(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return 100.0 * (after - before) / before


def pct_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def kernel_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in summary["kernels"]}


def compare(
    before: dict[str, Any], after: dict[str, Any], before_label: str, after_label: str
) -> dict[str, Any]:
    before_total = float(before["mean_total_kernel_ms_per_trace"])
    after_total = float(after["mean_total_kernel_ms_per_trace"])

    families = []
    family_names = set(before["family_ms"]) | set(after["family_ms"])
    for name in family_names:
        before_ms = float(before["family_ms"].get(name, 0.0))
        after_ms = float(after["family_ms"].get(name, 0.0))
        families.append(
            {
                "name": name,
                "before_ms": before_ms,
                "after_ms": after_ms,
                "delta_ms": after_ms - before_ms,
                "change_pct": pct_change(before_ms, after_ms),
            }
        )
    families.sort(key=lambda item: abs(item["delta_ms"]), reverse=True)

    before_kernels = kernel_map(before)
    after_kernels = kernel_map(after)
    kernels = []
    for name in set(before_kernels) | set(after_kernels):
        before_item = before_kernels.get(name, {})
        after_item = after_kernels.get(name, {})
        before_ms = float(before_item.get("mean_ms_per_trace", 0.0))
        after_ms = float(after_item.get("mean_ms_per_trace", 0.0))
        kernels.append(
            {
                "name": name,
                "family": after_item.get("family", before_item.get("family", "other")),
                "before_ms": before_ms,
                "after_ms": after_ms,
                "delta_ms": after_ms - before_ms,
                "change_pct": pct_change(before_ms, after_ms),
                "before_calls": float(before_item.get("mean_calls_per_trace", 0.0)),
                "after_calls": float(after_item.get("mean_calls_per_trace", 0.0)),
            }
        )
    kernels.sort(key=lambda item: abs(item["delta_ms"]), reverse=True)

    return {
        "before_label": before_label,
        "after_label": after_label,
        "before_total_ms": before_total,
        "after_total_ms": after_total,
        "total_delta_ms": after_total - before_total,
        "total_change_pct": pct_change(before_total, after_total),
        "families": families,
        "kernels": kernels,
    }


def print_comparison(result: dict[str, Any], top: int) -> None:
    print(
        f"{result['before_label']} -> {result['after_label']}: "
        f"{result['before_total_ms']:.3f} -> {result['after_total_ms']:.3f} ms/trace "
        f"({result['total_delta_ms']:+.3f} ms, "
        f"{pct_text(result['total_change_pct'])})"
    )
    print("\nFamily | before ms | after ms | delta ms | change")
    print("--- | ---: | ---: | ---: | ---:")
    for item in result["families"]:
        print(
            f"{item['name']} | {item['before_ms']:.3f} | {item['after_ms']:.3f} | "
            f"{item['delta_ms']:+.3f} | {pct_text(item['change_pct'])}"
        )

    print("\nKernel | family | before ms | after ms | delta ms | calls before/after")
    print("--- | --- | ---: | ---: | ---: | ---:")
    for item in result["kernels"][:top]:
        name = item["name"].replace("|", "\\|")
        print(
            f"{name} | {item['family']} | {item['before_ms']:.3f} | "
            f"{item['after_ms']:.3f} | {item['delta_ms']:+.3f} | "
            f"{item['before_calls']:.1f}/{item['after_calls']:.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--before-label")
    parser.add_argument("--after-label")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    result = compare(
        load(args.before),
        load(args.after),
        args.before_label or args.before.stem,
        args.after_label or args.after.stem,
    )
    print_comparison(result, args.top)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
