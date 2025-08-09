#!/usr/bin/env python3
"""
Plot additional metrics from aggregated stats JSON files.

This script mirrors the structure of `plot_aggregated_stats.py` but focuses on
metrics within each robot's `fitness_metrics` object, specifically:
 - MAX_DISTANCE
 - HEAD_STABILITY

For each aggregated experiment file in `aggregated_stats/*_aggregated.json`, it
creates per-generation plots of average, median, min, max with ±1 std band, as
well as median-with-quantiles plots for the two metrics.

Outputs are written under `aggregated_plots_metrics/<experiment_name>/`.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_aggregated_data(file_path: str) -> Dict[str, Any]:
    """Load aggregated data from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def _flatten(values: List[Any]) -> List[float]:
    """Flatten a list of scalars or lists into a flat list of floats, dropping None."""
    flat: List[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, list):
            for x in v:
                if x is None:
                    continue
                try:
                    flat.append(float(x))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                flat.append(float(v))
            except (TypeError, ValueError):
                continue
    return flat


def extract_metric_stats(
    data: Dict[str, Any], metric_key: str
) -> Dict[str, List[float]]:
    """Extract per-generation statistics for a metric in `fitness_metrics`.

    The metric values can be scalars or lists; None and non-numeric values are ignored.
    Returns a dict with generations, avg, std, median, min, max arrays.
    """
    generations: List[int] = []
    avg_values: List[float] = []
    std_values: List[float] = []
    median_values: List[float] = []
    min_values: List[float] = []
    max_values: List[float] = []

    robot_stats: Dict[str, List[Dict[str, Any]]] = data.get(
        "robot_stats_per_generation", {}
    )
    for gen in sorted(robot_stats.keys(), key=int):
        metric_vals_raw = [
            r.get("fitness_metrics", {}).get(metric_key, None) for r in robot_stats[gen]
        ]
        metric_vals = _flatten(metric_vals_raw)
        if len(metric_vals) == 0:
            continue

        arr = np.array(metric_vals, dtype=float)
        generations.append(int(gen))
        avg_values.append(float(np.mean(arr)))
        std_values.append(float(np.std(arr)))
        median_values.append(float(np.median(arr)))
        min_values.append(float(np.min(arr)))
        max_values.append(float(np.max(arr)))

    return {
        "generations": generations,
        "avg": avg_values,
        "std": std_values,
        "median": median_values,
        "min": min_values,
        "max": max_values,
    }


def extract_filtered_metric_stats(
    data: Dict[str, Any], metric_key: str, outlier_threshold: float = 2.0
) -> Dict[str, List[float]]:
    """Extract per-generation stats for a metric using z-score outlier filtering.

    For each generation, compute z-scores and keep values with |z| < threshold.
    If the standard deviation is zero, no filtering is applied for that generation.
    """
    generations: List[int] = []
    avg_values: List[float] = []
    std_values: List[float] = []
    median_values: List[float] = []
    min_values: List[float] = []
    max_values: List[float] = []

    robot_stats: Dict[str, List[Dict[str, Any]]] = data.get(
        "robot_stats_per_generation", {}
    )
    for gen in sorted(robot_stats.keys(), key=int):
        metric_vals_raw = [
            r.get("fitness_metrics", {}).get(metric_key, None) for r in robot_stats[gen]
        ]
        metric_vals = _flatten(metric_vals_raw)
        if len(metric_vals) == 0:
            continue

        arr = np.array(metric_vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std == 0.0:
            filtered = arr
        else:
            z = np.abs((arr - mean) / std)
            filtered = arr[z < outlier_threshold]

        if filtered.size == 0:
            continue

        generations.append(int(gen))
        avg_values.append(float(np.mean(filtered)))
        std_values.append(float(np.std(filtered)))
        median_values.append(float(np.median(filtered)))
        min_values.append(float(np.min(filtered)))
        max_values.append(float(np.max(filtered)))

    return {
        "generations": generations,
        "avg": avg_values,
        "std": std_values,
        "median": median_values,
        "min": min_values,
        "max": max_values,
    }


def compute_q1_q3_per_generation(
    data: Dict[str, Any], metric_key: str
) -> Tuple[List[int], List[float], List[float]]:
    """Compute Q1 and Q3 of a metric for each generation."""
    robot_stats: Dict[str, List[Dict[str, Any]]] = data.get(
        "robot_stats_per_generation", {}
    )
    generations: List[int] = []
    q1_values: List[float] = []
    q3_values: List[float] = []

    for gen in sorted(robot_stats.keys(), key=int):
        metric_vals_raw = [
            r.get("fitness_metrics", {}).get(metric_key, None) for r in robot_stats[gen]
        ]
        metric_vals = _flatten(metric_vals_raw)
        if len(metric_vals) == 0:
            continue
        arr = np.array(metric_vals, dtype=float)
        generations.append(int(gen))
        q1_values.append(float(np.percentile(arr, 25)))
        q3_values.append(float(np.percentile(arr, 75)))

    return generations, q1_values, q3_values


def create_metric_plot(
    stats_data: Dict[str, List[float]], output_path: Path, title: str, y_label: str
) -> None:
    """Create a comprehensive plot for a metric with avg±std, median, min, max."""
    if not stats_data["generations"]:
        return

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))

    generations = stats_data["generations"]

    # Avg ± 1 std
    plt.fill_between(
        generations,
        np.array(stats_data["avg"]) - np.array(stats_data["std"]),
        np.array(stats_data["avg"]) + np.array(stats_data["std"]),
        alpha=0.3,
        label="±1 Std Dev",
    )

    # Lines
    plt.plot(generations, stats_data["avg"], "b-", linewidth=2, label="Average")
    plt.plot(generations, stats_data["median"], "g-", linewidth=2, label="Median")
    plt.plot(generations, stats_data["min"], "r--", linewidth=1, label="Min")
    plt.plot(generations, stats_data["max"], "r-", linewidth=1, label="Max")

    plt.xlabel("Generation")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_median_with_quantiles_plot(
    stats_data: Dict[str, List[float]],
    raw_data: Dict[str, Any],
    metric_key: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    """Create a plot showing average/median with Q1-Q3 bounds for a metric."""
    if not stats_data["generations"]:
        return

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))

    generations = stats_data["generations"]
    avg_values = stats_data["avg"]

    # Per-generation Q1/Q3 from raw data
    gens_q, q1_values, q3_values = compute_q1_q3_per_generation(raw_data, metric_key)

    # Ensure alignment; if different (due to empty gens), intersect
    if gens_q != generations:
        # Build maps and realign
        q_map = {g: (q1, q3) for g, q1, q3 in zip(gens_q, q1_values, q3_values)}
        new_generations: List[int] = []
        new_avg: List[float] = []
        new_q1: List[float] = []
        new_q3: List[float] = []
        for g, a in zip(generations, avg_values):
            if g in q_map:
                q1, q3 = q_map[g]
                new_generations.append(g)
                new_avg.append(a)
                new_q1.append(q1)
                new_q3.append(q3)
        generations = new_generations
        avg_values = new_avg
        q1_values = new_q1
        q3_values = new_q3

    # Median line is not directly used; we show average to mirror other script
    plt.plot(generations, avg_values, "b-", linewidth=3, label="Average")
    plt.fill_between(generations, q1_values, q3_values, alpha=0.3, label="Q1-Q3 Range")

    plt.xlabel("Generation")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_aggregated_files() -> List[str]:
    """Find all aggregated JSON files in the aggregated_stats folder."""
    return glob.glob("aggregated_stats/*_aggregated.json")


def main() -> None:
    # Where to write plots
    output_root = Path("aggregated_plots_metrics")
    output_root.mkdir(exist_ok=True)

    aggregated_files = find_aggregated_files()
    print(f"Found {len(aggregated_files)} aggregated JSON files:")
    for file_path in aggregated_files:
        print(f"  - {file_path}")

    # Metric configuration: mapping from fitness_metrics key to (output stem, y label)
    metrics = {
        "MAX_DISTANCE": ("max_distance", "Max Distance"),
        "HEAD_STABILITY": ("head_stability", "Head Stability"),
    }

    for file_path in aggregated_files:
        print(f"\nProcessing {file_path}...")
        file_path_obj = Path(file_path)
        experiment_name = file_path_obj.stem.replace("_aggregated", "")
        experiment_output_dir = output_root / experiment_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = load_aggregated_data(file_path)
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
            continue

        for metric_key, (stem, y_label) in metrics.items():
            try:
                stats_data = extract_metric_stats(data, metric_key)
                if not stats_data["generations"]:
                    print(f"  Warning: No data for metric {metric_key} in {file_path}")
                    continue

                # Comprehensive per-generation plot
                plot_path = experiment_output_dir / f"{stem}_statistics.png"
                create_metric_plot(
                    stats_data,
                    plot_path,
                    f"{experiment_name} - {y_label} Statistics Over Time",
                    y_label,
                )
                print(f"  Created plot: {plot_path}")

                # Median with quantiles plot (using per-generation Q1/Q3)
                q_plot_path = (
                    experiment_output_dir / f"median_{stem}_with_quantiles.png"
                )
                create_median_with_quantiles_plot(
                    stats_data,
                    data,
                    metric_key,
                    q_plot_path,
                    f"{experiment_name} - Average {y_label} with Quantile Bounds",
                    y_label,
                )
                print(f"  Created plot: {q_plot_path}")

                # Outlier-filtered variants
                filtered_stats = extract_filtered_metric_stats(
                    data, metric_key, outlier_threshold=2.0
                )
                if filtered_stats["generations"]:
                    plot_path_f = (
                        experiment_output_dir / f"{stem}_statistics_filtered.png"
                    )
                    create_metric_plot(
                        filtered_stats,
                        plot_path_f,
                        f"{experiment_name} - {y_label} Statistics Over Time (Filtered)",
                        y_label,
                    )
                    print(f"  Created plot: {plot_path_f}")

                    q_plot_path_f = (
                        experiment_output_dir
                        / f"median_{stem}_with_quantiles_filtered.png"
                    )
                    create_median_with_quantiles_plot(
                        filtered_stats,
                        data,
                        metric_key,
                        q_plot_path_f,
                        f"{experiment_name} - Average {y_label} with Quantile Bounds (Filtered)",
                        y_label,
                    )
                    print(f"  Created plot: {q_plot_path_f}")

            except Exception as e:
                print(f"  Error processing metric {metric_key} for {file_path}: {e}")
                continue

    print(f"\nAll metric plots saved to: {output_root.absolute()}")


if __name__ == "__main__":
    main()
