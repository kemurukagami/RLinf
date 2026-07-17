"""Analyze a completed or interrupted collocated VLA/Wan timing run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("vla_onload_ms", "VLA onload"),
    ("vla_action_generation_ms", "VLA action generation"),
    ("vla_offload_ms", "VLA offload"),
    ("wm_onload_ms", "Wan onload"),
    ("wm_observation_generation_ms", "Wan observation generation"),
    ("wm_offload_ms", "Wan offload"),
    ("action_handoff_ms", "Action handoff"),
    ("observation_handoff_ms", "Observation handoff"),
    ("model_swap_ms", "All model transitions"),
    ("cycle_ms", "Complete cycle"),
)

COMPARABLE_METRICS = (
    ("vla_action_generation_ms", "vla_action_generation"),
    ("wm_observation_generation_ms", "wm_observation_generation"),
    ("action_handoff_ms", "action_handoff"),
    ("observation_handoff_ms", "observation_handoff"),
    ("cycle_ms", "cycle"),
)


def _read_rows(path: Path, limit: int) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)[:limit]
    if len(rows) != limit:
        raise ValueError(f"requested {limit} rows but {path} contains {len(rows)}")
    if [int(row["iteration"]) for row in rows] != list(range(limit)):
        raise ValueError("iteration identifiers are not contiguous from zero")
    for row in rows:
        for key, _ in METRICS:
            value = float(row[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {key} in iteration {row['iteration']}")
        expected_swap = sum(
            float(row[key])
            for key in (
                "vla_onload_ms",
                "vla_offload_ms",
                "wm_onload_ms",
                "wm_offload_ms",
            )
        )
        if not math.isclose(float(row["model_swap_ms"]), expected_swap, abs_tol=1e-6):
            raise ValueError(
                f"model-swap total mismatch in iteration {row['iteration']}"
            )
    return fieldnames, rows


def _summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "count": float(values.size),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


def _read_disaggregated_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["metric"]: {
                key: float(value)
                for key, value in row.items()
                if key != "metric" and value is not None
            }
            for row in csv.DictReader(stream)
        }


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, float]],
    comparisons: list[dict[str, Any]],
) -> None:
    x = np.arange(len(rows))
    figure, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(
        x, [float(row["vla_action_generation_ms"]) for row in rows], label="VLA"
    )
    axes[0].plot(
        x, [float(row["wm_observation_generation_ms"]) for row in rows], label="Wan"
    )
    axes[0].set_ylabel("Compute (ms)")
    axes[0].legend()
    for key, label in (
        ("vla_onload_ms", "VLA onload"),
        ("vla_offload_ms", "VLA offload"),
        ("wm_onload_ms", "Wan onload"),
        ("wm_offload_ms", "Wan offload"),
    ):
        axes[1].plot(x, [float(row[key]) for row in rows], label=label)
    axes[1].set_ylabel("Transition (ms)")
    axes[1].legend(ncol=2)
    axes[2].plot(x, [float(row["cycle_ms"]) for row in rows], label="Complete cycle")
    axes[2].set_xlabel("Measured iteration")
    axes[2].set_ylabel("Cycle (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"Collocated VLA/Wan latency across {len(rows)} iterations")
    figure.tight_layout()
    figure.savefig(output_dir / "latency_timeseries.png", dpi=180)
    plt.close(figure)

    compute = (
        summaries["vla_action_generation_ms"]["mean_ms"]
        + summaries["wm_observation_generation_ms"]["mean_ms"]
    )
    swap = summaries["model_swap_ms"]["mean_ms"]
    handoff = (
        summaries["action_handoff_ms"]["mean_ms"]
        + summaries["observation_handoff_ms"]["mean_ms"]
    )
    figure, axis = plt.subplots(figsize=(12, 4.5))
    left = 0.0
    for label, value, color in (
        ("Model compute", compute, "#4C78A8"),
        ("Model onload/offload", swap, "#F58518"),
        ("Payload handoff", handoff, "#54A24B"),
    ):
        axis.barh(["Mean cycle"], [value], left=left, label=label, color=color)
        fraction = value / (compute + swap + handoff)
        if fraction > 0.025:
            axis.text(
                left + value / 2,
                0,
                f"{value:.1f} ms",
                ha="center",
                va="center",
                rotation=90 if fraction < 0.1 else 0,
                fontsize=8 if fraction < 0.1 else None,
            )
        left += value
    axis.set_xlabel("Mean latency contribution (ms)")
    axis.tick_params(axis="y", left=False, labelleft=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    axis.set_title("Collocated cycle breakdown")
    figure.tight_layout()
    figure.savefig(output_dir / "mean_cycle_breakdown.png", dpi=180)
    plt.close(figure)

    positions = np.arange(len(comparisons))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        positions - width / 2,
        [row["disaggregated_mean_ms"] for row in comparisons],
        width,
        label="Disaggregated",
    )
    axis.bar(
        positions + width / 2,
        [row["collocated_mean_ms"] for row in comparisons],
        width,
        label="Collocated",
    )
    axis.set_yscale("log")
    axis.set_xticks(
        positions, [row["label"] for row in comparisons], rotation=18, ha="right"
    )
    axis.set_ylabel("Mean latency (ms, logarithmic scale)")
    axis.set_title("Collocated (180 iterations) vs disaggregated (300 iterations)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "collocated_vs_disaggregated.png", dpi=180)
    plt.close(figure)


def analyze(source: Path, disaggregated: Path, output_dir: Path, limit: int) -> None:
    fieldnames, rows = _read_rows(source, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "iterations.csv", tuple(fieldnames), rows)

    summaries = {key: _summary(rows, key) for key, _ in METRICS}
    summary_columns = ("metric", *next(iter(summaries.values())).keys())
    summary_rows = [
        {"metric": key.removesuffix("_ms"), **value} for key, value in summaries.items()
    ]
    _write_csv(output_dir / "summary.csv", summary_columns, summary_rows)

    disaggregated_summary = _read_disaggregated_summary(disaggregated)
    labels = dict(METRICS)
    comparisons = []
    for collocated_key, disaggregated_key in COMPARABLE_METRICS:
        collocated_mean = summaries[collocated_key]["mean_ms"]
        disaggregated_mean = disaggregated_summary[disaggregated_key]["mean_ms"]
        comparisons.append(
            {
                "metric": collocated_key.removesuffix("_ms"),
                "label": labels[collocated_key],
                "disaggregated_mean_ms": disaggregated_mean,
                "collocated_mean_ms": collocated_mean,
                "collocated_minus_disaggregated_ms": collocated_mean
                - disaggregated_mean,
                "collocated_over_disaggregated": collocated_mean / disaggregated_mean,
            }
        )
    _write_csv(
        output_dir / "collocated_vs_disaggregated.csv",
        tuple(comparisons[0].keys()),
        comparisons,
    )

    midpoint = limit // 2
    stability_rows = []
    for key, label in METRICS:
        first = _summary(rows[:midpoint], key)["mean_ms"]
        second = _summary(rows[midpoint:], key)["mean_ms"]
        stability_rows.append(
            {
                "metric": key.removesuffix("_ms"),
                "label": label,
                "first_half_mean_ms": first,
                "second_half_mean_ms": second,
                "second_vs_first_percent": 100 * (second / first - 1),
            }
        )
    _write_csv(
        output_dir / "stability.csv", tuple(stability_rows[0].keys()), stability_rows
    )

    cycle = summaries["cycle_ms"]["mean_ms"]
    swap = summaries["model_swap_ms"]["mean_ms"]
    compute = (
        summaries["vla_action_generation_ms"]["mean_ms"]
        + summaries["wm_observation_generation_ms"]["mean_ms"]
    )
    handoff = (
        summaries["action_handoff_ms"]["mean_ms"]
        + summaries["observation_handoff_ms"]["mean_ms"]
    )
    disaggregated_cycle = disaggregated_summary["cycle"]["mean_ms"]
    derived = [
        {
            "metric": "model_swap_share_of_cycle",
            "value": 100 * swap / cycle,
            "unit": "percent",
        },
        {
            "metric": "compute_share_of_cycle",
            "value": 100 * compute / cycle,
            "unit": "percent",
        },
        {
            "metric": "handoff_share_of_cycle",
            "value": 100 * handoff / cycle,
            "unit": "percent",
        },
        {
            "metric": "action_chunks_per_second",
            "value": 1000 / cycle,
            "unit": "chunks/s",
        },
        {
            "metric": "environment_actions_per_second",
            "value": 8000 / cycle,
            "unit": "actions/s",
        },
        {
            "metric": "collocated_over_disaggregated_cycle",
            "value": cycle / disaggregated_cycle,
            "unit": "ratio",
        },
    ]
    _write_csv(output_dir / "derived_metrics.csv", ("metric", "value", "unit"), derived)
    _write_plots(output_dir, rows, summaries, comparisons)

    report = f"""# Collocated OpenVLA-OFT + Wan: {limit}-iteration analysis

This report freezes iterations 0–{limit - 1} from the interrupted collocated run.
All rows passed finite-positive, contiguous-index, and model-swap-total checks.

| Metric | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
"""
    for key, label in METRICS:
        item = summaries[key]
        report += (
            f"| {label} | {item['mean_ms']:.3f} | {item['p50_ms']:.3f} | "
            f"{item['p95_ms']:.3f} | {item['p99_ms']:.3f} | {item['max_ms']:.3f} |\n"
        )
    report += f"""
- Model-transition share: **{100 * swap / cycle:.2f}%**
- Compute share: **{100 * compute / cycle:.2f}%**
- Payload-handoff share: **{100 * handoff / cycle:.4f}%**
- Throughput: **{1000 / cycle:.4f} chunks/s**, or **{8000 / cycle:.4f} environment actions/s**
- Collocated/disaggregated mean-cycle ratio: **{cycle / disaggregated_cycle:.2f}×**

The benchmark uses one environment, an 8×7 action chunk, one Wan inference
step, one shared RTX 4090, CUDA synchronization at timing boundaries, and
includes allocator cache release in offload measurements. It excludes Ray and
RLinf Channel overhead.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    metadata = {
        "source": str(source),
        "source_rows_available": sum(1 for _ in source.open(encoding="utf-8")) - 1,
        "rows_analyzed": limit,
        "first_iteration": 0,
        "last_iteration": limit - 1,
        "disaggregated_summary": str(disaggregated),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--disaggregated-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    args = parser.parse_args()
    analyze(
        args.source.resolve(),
        args.disaggregated_summary.resolve(),
        args.output_dir.resolve(),
        args.limit,
    )


if __name__ == "__main__":
    main()
