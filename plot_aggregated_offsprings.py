#!/usr/bin/env python3
"""
Plot average number of offsprings per generation from aggregated offspring JSON files.

The input files live in `aggregated_stats/` and end with `_aggregated_offsprings.json`.
Each file is an aggregation of 5 runs, and `offspring_count` is counted for both
parents. Therefore, total offspring per generation must be corrected by dividing by 2,
and then divided by the number of runs (default 5) to get a per-run average.

Outputs are written under `aggregated_offsprings/<experiment_name>/` as
`average_offsprings_per_generation.png`.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_offspring_data(file_path: str) -> Dict[str, Any]:
    """Load aggregated offspring data from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_average_offsprings_per_generation(
    data: Dict[str, Any], runs_per_aggregate: int = 5
) -> Tuple[List[int], List[float]]:
    """Compute per-generation average number of offsprings per run.

    Expects `data` to map generation keys (str) to a list of robot dicts, each having
    an `offspring_count` attribute representing the number of offspring that robot
    contributed to. Offspring counts are double-counted (once per parent), so we first
    divide by 2, and then divide by `runs_per_aggregate` to get the per-run average.
    """
    generations: List[int] = []
    avg_offsprings_per_run: List[float] = []

    # Some files might store a nested object; accept top-level mapping as primary
    # Fall back to `data.get("per_generation", data)` if needed
    per_generation: Dict[str, Any] = data if isinstance(data, dict) else {}
    per_generation = per_generation.get("per_generation", per_generation)

    for gen in sorted(per_generation.keys(), key=int):
        entries = per_generation.get(gen, [])
        total_offspring_double_counted = 0
        for entry in entries:
            count = entry.get("offspring_count", 0)
            if isinstance(count, (int, float)):
                total_offspring_double_counted += float(count)

        # Correct double counting (one offspring counts for both parents)
        total_offspring = total_offspring_double_counted / 2.0
        # Convert to per-run average using known aggregation size
        avg_per_run = total_offspring / max(1, runs_per_aggregate)
        generations.append(int(gen))
        avg_offsprings_per_run.append(avg_per_run)

    return generations, avg_offsprings_per_run


def compute_total_unique_producers_per_run(
    data: Dict[str, Any], runs_per_aggregate: int = 5
) -> Tuple[int, float]:
    """Compute per-run statistic of unique producers across the whole experiment.

    Count unique robotIds that have offspring_count > 0 in any generation, then divide
    by `runs_per_aggregate` (default 5).
    """
    per_generation: Dict[str, Any] = data if isinstance(data, dict) else {}
    per_generation = per_generation.get("per_generation", per_generation)

    unique_ids = set()
    for gen in per_generation.keys():
        if gen == "399":
            continue
        entries = per_generation.get(gen, [])
        for entry in entries:
            cnt = entry.get("offspring_count", 0)
            rid = entry.get("robotId")
            try:
                produced = float(cnt) > 0.0
            except (TypeError, ValueError):
                produced = False
            if produced and rid is not None:
                unique_ids.add(rid)

    total_unique = len(unique_ids)
    per_run_unique = total_unique / max(1, runs_per_aggregate)
    return total_unique, per_run_unique


def save_unique_producers_summary(
    total_unique: int, per_run_unique: float, output_path: Path, experiment_name: str
) -> None:
    """Save overall per-run unique producers summary (total unique / 5)."""
    lines = [
        f"Experiment: {experiment_name}",
        f"Total unique producers across all generations (aggregated): {total_unique}",
        f"Per-run unique producers (total_unique / 5): {per_run_unique:.3f}",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def create_average_offsprings_plot(
    generations: List[int],
    avg_offsprings: List[float],
    output_path: Path,
    title: str,
) -> None:
    """Plot average number of offsprings per generation (per run)."""
    if not generations:
        return

    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))
    plt.plot(
        generations, avg_offsprings, linewidth=2, label="Average Offsprings (per run)"
    )
    plt.xlabel("Generation")
    plt.ylabel("Offsprings per Generation (per run)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_aggregated_offspring_files() -> List[str]:
    """Find aggregated offspring JSON files in the aggregated_stats folder."""
    return glob.glob("aggregated_stats/*_aggregated_offsprings.json")


def create_combined_average_birth_rate_plot(
    aggregated_files: List[str],
    output_path: Path,
    fitness_key: str,
    runs_per_aggregate: int = 5,
) -> None:
    """Create a combined plot of average birth rate per generation for two death mechanisms.

    Compares `lowest_fitness` and `max_age` for the provided fitness key
    (one of: "combo", "max_head", "max_npl"). Uses per-run births per generation.
    """
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))

    cases = [f"lowest_fitness_{fitness_key}", f"max_age_{fitness_key}"]
    labels = ["Lowest Fitness", "Max Age"]
    colors = ["#1f77b4", "#d62728"]  # Blue, Red

    any_series_plotted = False

    for i, case in enumerate(cases):
        case_file = None
        for file_path in aggregated_files:
            if file_path.endswith("_aggregated_offsprings.json") and case in file_path:
                case_file = file_path
                break
        if case_file is None:
            print(f"Warning: Could not find offspring file for case {case}")
            continue

        try:
            data = load_offspring_data(case_file)
            generations, avg_offsprings = extract_average_offsprings_per_generation(
                data, runs_per_aggregate=runs_per_aggregate
            )
            if len(generations) == 0:
                continue

            plt.plot(
                generations,
                avg_offsprings,
                linewidth=2,
                label=labels[i],
                color=colors[i],
            )
            any_series_plotted = True
        except Exception as e:
            print(f"Error processing {case_file}: {e}")
            continue

    if not any_series_plotted:
        plt.close()
        return

    title_map = {
        "combo": "Combo",
        "max_head": "Max Head Stability",
        "max_npl": "Max NPL",
    }
    pretty_key = title_map.get(fitness_key, fitness_key)

    plt.xlabel("Generation")
    plt.ylabel("Births per Generation (per run)")
    plt.title(f"Average Birth Rate per Generation - {pretty_key}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    output_root = Path("aggregated_offsprings")
    output_root.mkdir(exist_ok=True)

    files = find_aggregated_offspring_files()
    print(f"Found {len(files)} aggregated offspring JSON files:")
    for fp in files:
        print(f"  - {fp}")

    for file_path in files:
        print(f"\nProcessing {file_path}...")
        file_path_obj = Path(file_path)
        experiment_name = file_path_obj.stem.replace("_aggregated_offsprings", "")
        experiment_output_dir = output_root / experiment_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            data = load_offspring_data(file_path)
            generations, avg_offsprings = extract_average_offsprings_per_generation(
                data, runs_per_aggregate=5
            )
            if not generations:
                print("  Warning: no generations found")
                continue

            output_path = (
                experiment_output_dir / "average_offsprings_per_generation.png"
            )
            create_average_offsprings_plot(
                generations,
                avg_offsprings,
                output_path,
                f"{experiment_name} - Average Offsprings per Generation (per run)",
            )
            print(f"  Created plot: {output_path}")

            # Compute and save unique producers summary (overall per-run statistic)
            total_unique, per_run_unique = compute_total_unique_producers_per_run(
                data, runs_per_aggregate=5
            )
            summary_path = experiment_output_dir / "unique_producers_summary.txt"
            save_unique_producers_summary(
                total_unique, per_run_unique, summary_path, experiment_name
            )
            print(f"  Created summary: {summary_path}")
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue

    # Create combined average birth rate plots (per run)
    print(
        "\nCreating combined average birth rate plots (per run, 5 runs aggregated)..."
    )
    avg_birth_combo_path = output_root / "average_birth_rate_combo.png"
    avg_birth_max_head_path = output_root / "average_birth_rate_max_head.png"
    avg_birth_max_npl_path = output_root / "average_birth_rate_max_npl.png"

    create_combined_average_birth_rate_plot(
        files, avg_birth_combo_path, fitness_key="combo", runs_per_aggregate=5
    )
    print("Created combined average birth rate plot:", avg_birth_combo_path)

    create_combined_average_birth_rate_plot(
        files, avg_birth_max_head_path, fitness_key="max_head", runs_per_aggregate=5
    )
    print("Created combined average birth rate plot:", avg_birth_max_head_path)

    create_combined_average_birth_rate_plot(
        files, avg_birth_max_npl_path, fitness_key="max_npl", runs_per_aggregate=5
    )
    print("Created combined average birth rate plot:", avg_birth_max_npl_path)

    print(f"\nAll offspring plots saved to: {output_root.absolute()}")


if __name__ == "__main__":
    main()
