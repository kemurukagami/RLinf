"""Create spreadsheet and plot comparisons for VLA/Wan timing benchmarks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = (
    ("vla_action_generation", "VLA action generation"),
    ("wm_observation_generation", "Wan observation generation"),
    ("action_handoff", "Action handoff"),
    ("observation_handoff", "Observation handoff"),
    ("cycle", "Complete cycle"),
)


def _read_summary(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row["metric"]: {
                key: float(value)
                for key, value in row.items()
                if key != "metric" and value is not None
            }
            for row in csv.DictReader(stream)
        }


def compare(disaggregated: Path, collocated: Path, output_dir: Path) -> None:
    disaggregated_summary = _read_summary(disaggregated / "summary.csv")
    collocated_summary = _read_summary(collocated / "summary.csv")
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = (
        "metric",
        "label",
        "disaggregated_mean_ms",
        "collocated_mean_ms",
        "collocated_minus_disaggregated_ms",
        "collocated_over_disaggregated",
        "disaggregated_p95_ms",
        "collocated_p95_ms",
        "disaggregated_p99_ms",
        "collocated_p99_ms",
    )
    rows = []
    for metric, label in METRICS:
        disaggregated_row = disaggregated_summary[metric]
        collocated_row = collocated_summary[metric]
        disaggregated_mean = disaggregated_row["mean_ms"]
        collocated_mean = collocated_row["mean_ms"]
        rows.append(
            {
                "metric": metric,
                "label": label,
                "disaggregated_mean_ms": disaggregated_mean,
                "collocated_mean_ms": collocated_mean,
                "collocated_minus_disaggregated_ms": (
                    collocated_mean - disaggregated_mean
                ),
                "collocated_over_disaggregated": (collocated_mean / disaggregated_mean),
                "disaggregated_p95_ms": disaggregated_row["p95_ms"],
                "collocated_p95_ms": collocated_row["p95_ms"],
                "disaggregated_p99_ms": disaggregated_row["p99_ms"],
                "collocated_p99_ms": collocated_row["p99_ms"],
            }
        )

    with (output_dir / "collocated_vs_disaggregated.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["label"] for row in rows]
    disaggregated_values = [row["disaggregated_mean_ms"] for row in rows]
    collocated_values = [row["collocated_mean_ms"] for row in rows]
    positions = list(range(len(rows)))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        [position - width / 2 for position in positions],
        disaggregated_values,
        width,
        label="Disaggregated",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        collocated_values,
        width,
        label="Collocated",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Mean latency (ms, logarithmic scale)")
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_title("OpenVLA-OFT + Wan: collocated vs disaggregated latency")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "collocated_vs_disaggregated.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disaggregated", type=Path, required=True)
    parser.add_argument("--collocated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(
        args.disaggregated.resolve(),
        args.collocated.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
