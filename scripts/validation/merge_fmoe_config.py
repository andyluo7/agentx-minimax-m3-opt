"""Merge AITER's shipped FMoE tables with a custom tuned table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aiter-root", required=True, type=Path)
    parser.add_argument("--custom", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config_roots = (
        args.aiter_root / "aiter" / "configs",
        args.aiter_root / "configs",
    )
    config_root = next((path for path in config_roots if path.is_dir()), None)
    if config_root is None:
        searched = ", ".join(str(path) for path in config_roots)
        raise FileNotFoundError(f"Could not find AITER configs under: {searched}")
    paths = [config_root / "tuned_fmoe.csv"]
    paths.extend(
        path
        for path in sorted((config_root / "model_configs").glob("*tuned_fmoe*.csv"))
        if "untuned" not in path.name
    )
    paths.append(args.custom)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing FMoE config files: {missing}")

    frames = [pd.read_csv(path) for path in paths]
    columns = list(frames[0].columns)
    for frame in frames[1:]:
        for column in frame.columns:
            if column not in columns:
                position = (
                    columns.index("tflops") if "tflops" in columns else len(columns)
                )
                columns.insert(position, column)

    defaults = {"xbf16": 0, "run_1stage": 0, "ksplit": 0, "_tag": ""}
    normalized = []
    for frame in frames:
        frame = frame.copy()
        for column in columns:
            if column not in frame.columns:
                if column == "gfx" and "cu_num" in frame.columns:
                    frame[column] = frame["cu_num"].map(
                        {256: "gfx950", 304: "gfx942", 80: "gfx942"}
                    )
                else:
                    frame[column] = defaults.get(column, 0)
        normalized.append(frame[columns])

    merged = pd.concat(normalized, ignore_index=True)
    if "_tag" in merged.columns:
        merged["_tag"] = merged["_tag"].fillna("")
    key_columns = list(pd.read_csv(config_root / "untuned_fmoe.csv", nrows=0).columns)
    for column in ("cu_num", "gfx", "_tag"):
        if column in merged.columns and column not in key_columns:
            key_columns.append(column)
    merged["us"] = pd.to_numeric(merged["us"], errors="coerce")
    merged = (
        merged.sort_values("us", kind="stable", na_position="last")
        .drop_duplicates(subset=key_columns, keep="first")
        .reset_index(drop=True)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(
        f"merged_files={len(paths)} rows={len(merged)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
