#!/usr/bin/env python3
"""Build the MiniMax-M3 AgentX MI355X/B200 Pareto comparison."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from verify_70pct import canonical_agentx_status, profiled_error_count


BENCHMARKS_URL = (
    "https://inferencex.semianalysis.com/api/v1/benchmarks?model=MiniMax-M3"
)
DERIVED_URL = (
    "https://inferencex.semianalysis.com/api/v1/derived-agentic-metrics?ids={}"
)
PUBLIC_MI355X_RUN = "31558297538"
B200_RUN = "31833401868"

DEFAULT_CANONICAL_ROOT = Path(__file__).with_name("artifacts") / "canonical-k4-v19-20260822"
DEFAULT_SWEEP_ROOT = Path(__file__).with_name("artifacts") / "b200-matched-curve-20260823-r1"
DEFAULT_OUTPUT_PREFIX = Path.home() / "Downloads" / "minimax-m3-agentx-pareto-mi355x-vs-b200"


def normalize_backend(value: Any) -> str | None:
    """Normalize backend metadata emitted as either a string or an object."""
    if isinstance(value, dict):
        value = value.get("name")
    return str(value) if value else None


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": "minimax-m3-agentx-pareto/2.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        if response.headers.get("Content-Encoding") == "gzip":
            payload = gzip.decompress(payload)
    return json.loads(payload)


def choose_public_variant(hardware: str, conc: int, candidates: list[dict[str, Any]]):
    """Match the published curves: resident at low C, DRAM offload at high C."""
    prefer_offload = (hardware == "b200" and conc >= 15) or (
        hardware == "mi355x" and conc >= 32
    )
    preferred = [
        point
        for point in candidates
        if (point["offload_mode"] == "on") == prefer_offload
    ]
    pool = preferred or candidates
    return max(pool, key=lambda point: (point["throughput"], point["interactivity"]))


def load_public_series() -> dict[str, list[dict[str, Any]]]:
    rows = fetch_json(BENCHMARKS_URL)
    rows = [
        row
        for row in rows
        if row.get("model") == "minimaxm3"
        and row.get("benchmark_type") == "agentic_traces"
        and row.get("hardware") in {"mi355x", "b200"}
        and row.get("framework") == "vllm"
        and row.get("precision") == "fp4"
        and row.get("spec_method") == "mtp"
        and row.get("disagg") is False
        and row.get("prefill_tp") == 4
        and row.get("decode_tp") == 4
        and (
            (row.get("hardware") == "mi355x" and PUBLIC_MI355X_RUN in (row.get("run_url") or ""))
            or (row.get("hardware") == "b200" and B200_RUN in (row.get("run_url") or ""))
        )
    ]
    ids = ",".join(str(row["id"]) for row in rows)
    derived = fetch_json(DERIVED_URL.format(ids))

    by_hardware: dict[str, list[dict[str, Any]]] = {"mi355x": [], "b200": []}
    for row in rows:
        extra = derived.get(str(row["id"]), {})
        interactivity = extra.get("p90_e2e_norm_intvty")
        throughput = row.get("metrics", {}).get("tput_per_gpu")
        if interactivity is None or throughput is None:
            continue
        by_hardware[row["hardware"]].append(
            {
                "series": "Current public MI355X" if row["hardware"] == "mi355x" else "B200 reference",
                "hardware": row["hardware"],
                "conc": int(row["conc"]),
                "throughput": float(throughput),
                "interactivity": float(interactivity),
                "offload_mode": row.get("offload_mode", "off"),
                "offload_backend": normalize_backend(
                    row.get("metrics", {}).get("kv_offload_backend")
                ),
                "source": row.get("run_url", ""),
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {}
    for hardware, points in by_hardware.items():
        chosen = []
        for conc in sorted({point["conc"] for point in points}):
            candidates = [point for point in points if point["conc"] == conc]
            chosen.append(choose_public_variant(hardware, conc, candidates))
        selected[hardware] = chosen
    return selected


def load_metadata(directory: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    path = directory / "run-metadata.txt"
    if not path.is_file():
        return metadata
    for line in path.read_text(errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            metadata[key] = value
    return metadata


def load_new_points(roots: list[Path]) -> list[dict[str, Any]]:
    points: dict[tuple[int, int, str], dict[str, Any]] = {}
    invalid: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            try:
                data = json.loads(path.read_text())
                throughput = data["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]
                interactivity = data["request_metrics"]["latency"]["e2e_norm_intvty"]["p90"]
                raw = data["request_metrics"]["latency"]["intvty"]["p90"]
                tpot = data["request_metrics"]["latency"]["tpot"]["p90"]
                conc = int(data["conc"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if "mi355x" not in str(data.get("hw", "")).lower():
                continue
            if str(data.get("infmax_model_prefix", "")).lower() != "minimaxm3":
                continue

            accounting = data.get("request_accounting", {})
            aggregate_errors = int(accounting.get("records_error_dropped", 0))
            measured_errors, error_source = profiled_error_count(path, aggregate_errors)
            canonical, reasons, jit_events, jit_source, replay_rc, replay_source = (
                canonical_agentx_status(path)
            )
            successful = int(data.get("num_requests_successful", 0))
            if successful <= 0 or measured_errors or not canonical:
                invalid.append(
                    f"{path}: successful={successful}, measured_errors={measured_errors} "
                    f"({error_source}), canonical={canonical}, reasons={reasons}"
                )
                continue

            metadata = load_metadata(path.parent)
            try:
                tp = int(data.get("tp") or metadata.get("tp") or 0)
            except (TypeError, ValueError):
                tp = 0
            if tp not in {4, 8}:
                invalid.append(f"{path}: tp={tp}, expected TP4 or TP8")
                continue
            offload = str(data.get("kv_offloading") or metadata.get("kv_offloading") or "none")
            backend = normalize_backend(
                data.get("kv_offload_backend") or metadata.get("kv_offload_backend")
            )
            variant = "resident" if offload == "none" else str(backend or offload)
            key = (conc, tp, variant)
            points[key] = {
                "series": "New MI355X",
                "hardware": "mi355x",
                "conc": conc,
                "tp": tp,
                "throughput": float(throughput),
                "interactivity": float(interactivity),
                "raw_interactivity": float(raw),
                "p90_tpot_ms": float(tpot) * 1000.0,
                "offload_mode": "off" if offload == "none" else "on",
                "offload_backend": backend,
                "successful_requests": successful,
                "measured_errors": measured_errors,
                "post_profile_jit_events": jit_events,
                "jit_source": jit_source,
                "replay_rc": replay_rc,
                "replay_source": replay_source,
                "source": str(path),
            }
    if invalid:
        print("Skipped invalid/noncanonical aggregate candidates:")
        for message in invalid:
            print(f"  {message}")
    return sorted(
        points.values(),
        key=lambda point: (point["conc"], point["tp"], point["offload_mode"]),
    )


def primary_new_curve(points: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    curve = []
    alternates = []
    for conc in sorted({point["conc"] for point in points}):
        candidates = [point for point in points if point["conc"] == conc]
        offloaded = [
            point
            for point in candidates
            if point["offload_mode"] == "on" and point["tp"] == 4
        ]
        resident_tp4 = [
            point
            for point in candidates
            if point["offload_mode"] == "off" and point["tp"] == 4
        ]
        preferred = offloaded if conc >= 15 and offloaded else resident_tp4
        selected = max(
            preferred or candidates,
            key=lambda point: (point["throughput"], point["interactivity"]),
        )
        curve.append(selected)
        alternates.extend(point for point in candidates if point is not selected)
    return curve, alternates


def write_csv(path: Path, groups: list[list[dict[str, Any]]]) -> None:
    fields = [
        "series",
        "hardware",
        "concurrency",
        "tensor_parallel",
        "kv_offload",
        "p90_e2e_normalized_interactivity_tok_s_user",
        "p90_raw_interactivity_tok_s_user",
        "p90_tpot_ms",
        "total_throughput_tok_s_chip",
        "successful_requests",
        "measured_errors",
        "post_profile_jit_events",
        "source",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for points in groups:
            for point in points:
                writer.writerow(
                    {
                        "series": point["series"],
                        "hardware": point["hardware"],
                        "concurrency": point["conc"],
                        "tensor_parallel": point.get("tp", 4),
                        "kv_offload": point.get("offload_backend") or point.get("offload_mode", "off"),
                        "p90_e2e_normalized_interactivity_tok_s_user": f'{point["interactivity"]:.8f}',
                        "p90_raw_interactivity_tok_s_user": (
                            f'{point["raw_interactivity"]:.8f}'
                            if "raw_interactivity" in point
                            else ""
                        ),
                        "p90_tpot_ms": (
                            f'{point["p90_tpot_ms"]:.8f}'
                            if "p90_tpot_ms" in point
                            else ""
                        ),
                        "total_throughput_tok_s_chip": f'{point["throughput"]:.8f}',
                        "successful_requests": point.get("successful_requests", ""),
                        "measured_errors": point.get("measured_errors", ""),
                        "post_profile_jit_events": point.get("post_profile_jit_events", ""),
                        "source": point["source"],
                    }
                )


def fmt_k(value: float, _position: float | None = None) -> str:
    return "0" if value == 0 else f"{value / 1000:.0f}K"


def plot(points: list[dict[str, Any]], output_prefix: Path) -> None:
    public = load_public_series()
    current = public["mi355x"]
    b200 = public["b200"]
    new_curve, alternates = primary_new_curve(points)
    if not new_curve:
        raise SystemExit("No validated new MI355X aggregate points found")

    colors = {
        "new": "#16A34A",
        "current": "#F59E0B",
        "b200": "#2563EB",
        "ink": "#172033",
        "muted": "#667085",
        "grid": "#DCE3EC",
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": colors["ink"],
            "axes.edgecolor": "#AAB4C3",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(16.8, 9.4), facecolor="#F8FAFC")
    ax.set_facecolor("white")

    def draw_line(series: list[dict[str, Any]], color: str, marker: str = "o") -> None:
        ax.plot(
            [point["interactivity"] for point in series],
            [point["throughput"] for point in series],
            color=color,
            linewidth=2.5,
            alpha=0.9,
            marker=marker,
            markersize=7,
            markerfacecolor="white" if marker == "o" else color,
            markeredgecolor=color,
            markeredgewidth=2,
            zorder=4,
        )

    draw_line(current, colors["current"])
    draw_line(b200, colors["b200"])
    draw_line(new_curve, colors["new"], "D")
    if alternates:
        ax.scatter(
            [point["interactivity"] for point in alternates],
            [point["throughput"] for point in alternates],
            s=100,
            marker="D",
            facecolors="white",
            edgecolors=colors["new"],
            linewidths=2.2,
            zorder=6,
        )

    label_concs = {1, 5, 10, 15, 20, 25, 28, 30, 32}
    for series, color in ((current, colors["current"]), (b200, colors["b200"])):
        for point in series:
            if point["conc"] not in label_concs:
                continue
            ax.annotate(
                f'C{point["conc"]}',
                (point["interactivity"], point["throughput"]),
                xytext=(8, 0) if point["interactivity"] < 70 else (0, 9),
                textcoords="offset points",
                color=color,
                fontsize=9,
                zorder=8,
            )
    new_label_offsets = {
        25: (8, -18),
        30: (8, 14),
        32: (-8, 14),
    }
    new_label_alignments = {32: "right"}
    for point in new_curve:
        suffix = " offload" if point["offload_mode"] == "on" else ""
        ax.annotate(
            f'C{point["conc"]} TP{point["tp"]}{suffix}',
            (point["interactivity"], point["throughput"]),
            xytext=new_label_offsets.get(point["conc"], (7, 8)),
            textcoords="offset points",
            color="#116329",
            fontsize=9,
            fontweight="bold",
            ha=new_label_alignments.get(point["conc"], "left"),
            zorder=9,
        )
    for point in alternates:
        suffix = " offload" if point["offload_mode"] == "on" else " resident"
        ax.annotate(
            f'C{point["conc"]} TP{point["tp"]}{suffix}',
            (point["interactivity"], point["throughput"]),
            xytext=(10, 12),
            textcoords="offset points",
            color="#116329",
            fontsize=9,
            zorder=9,
        )

    x_max = max(point["interactivity"] for group in (current, b200, points) for point in group)
    y_max = max(point["throughput"] for group in (current, b200, points) for point in group)
    ax.set_xlim(0, max(135, x_max * 1.08))
    ax.set_ylim(0, max(50000, y_max * 1.08))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_locator(MultipleLocator(10000))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_k))
    ax.grid(which="major", color=colors["grid"], linewidth=1.0, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(
        "P90 E2E normalized interactivity (tokens/s/user)  →",
        fontsize=13,
        fontweight="bold",
        labelpad=13,
    )
    ax.set_ylabel(
        "Total throughput per chip (tokens/s)  →",
        fontsize=13,
        fontweight="bold",
        labelpad=13,
    )

    fig.suptitle(
        "MiniMax-M3 AgentX Pareto Comparison",
        x=0.065,
        y=0.965,
        ha="left",
        fontsize=25,
        fontweight="bold",
        color=colors["ink"],
    )
    tp8_suffix = " + TP8 C1" if any(point.get("tp") == 8 for point in points) else ""
    fig.text(
        0.065,
        0.922,
        "New MI355X vs current public MI355X and matched B200 · "
        f"FP4 · vLLM/MTP · TP4 curve{tp8_suffix} · higher is better ↗",
        ha="left",
        fontsize=12.5,
        color=colors["muted"],
    )
    handles = [
        Line2D([0], [0], color=colors["new"], marker="D", markersize=8, linewidth=2.5,
               label="New MI355X · validated"),
        Line2D([0], [0], color=colors["current"], marker="o", markerfacecolor="white",
               markeredgewidth=2, linewidth=2.5, label="Current public MI355X · vLLM TP4"),
        Line2D([0], [0], color=colors["b200"], marker="o", markerfacecolor="white",
               markeredgewidth=2, linewidth=2.5, label="B200 reference · vLLM TP4"),
    ]
    if alternates:
        handles.insert(
            1,
            Line2D([0], [0], color=colors["new"], marker="D", markerfacecolor="white",
                   markeredgewidth=2, linewidth=0, label="New MI355X · alternate policy/TP"),
        )
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#D0D5DD",
        framealpha=0.96,
        fontsize=10.5,
        borderpad=0.9,
        labelspacing=0.8,
    )
    fig.text(
        0.065,
        0.018,
        f"Sources: InferenceX public AgentX runs {PUBLIC_MI355X_RUN} and {B200_RUN}; "
        "new MI355X validated aggregate artifacts. Offloaded points use vllm-simple DRAM KV offload.",
        ha="left",
        fontsize=9.3,
        color=colors["muted"],
    )
    fig.subplots_adjust(left=0.085, right=0.965, top=0.865, bottom=0.13)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    csv_path = output_prefix.with_suffix(".csv")
    fig.savefig(png_path, dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, facecolor=fig.get_facecolor())
    write_csv(csv_path, [new_curve, alternates, current, b200])
    print(png_path)
    print(pdf_path)
    print(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        dest="artifact_roots",
        help="Root containing validated MI355X aggregate artifacts; repeatable.",
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    roots = args.artifact_roots or [DEFAULT_CANONICAL_ROOT, DEFAULT_SWEEP_ROOT]
    plot(load_new_points(roots), args.output_prefix)


if __name__ == "__main__":
    main()
