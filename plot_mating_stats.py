#!/usr/bin/env python3
"""
Script to plot mating statistics from run1_offsprings_corrected.json files for each experiment.
For each experiment, creates a plot of the number of matings per generation.
For non-standard setups, overlays the mating/meeting rate per generation as well.

Usage:
    python plot_mating_stats.py --all-stats

This will search the stats/ directory for all run1_offsprings_corrected.json files and output plots to run1_plots_mating/ mirroring the stats/ structure.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from pathlib import Path


def load_data(filename):
    with open(filename, "r") as f:
        return json.load(f)


def is_standard_setup(path):
    # Returns True if the path is under stats/standard_setup/
    return "standard_setup" in Path(path).parts


def calculate_matings_per_generation(offsprings):
    """Returns a dict: generation -> number of matings in that generation."""
    matings_per_gen = {}
    for robot_id, robot_data in offsprings.items():
        for gen_str, gen_data in robot_data.items():
            gen = int(gen_str)
            if gen not in matings_per_gen:
                matings_per_gen[gen] = 0
            # Count meetings as matings
            matings_per_gen[gen] += gen_data.get("meeting_count", 0)

    # Divide by 2 to account for double-counting (each mating involves two robots)
    for gen in matings_per_gen:
        matings_per_gen[gen] = matings_per_gen[gen] // 2

    return matings_per_gen


def calculate_meeting_rate_per_generation(offsprings):
    """Returns a dict: generation -> average meeting rate (float) for that generation."""
    meeting_rate_per_gen = {}
    for robot_id, robot_data in offsprings.items():
        for gen_str, gen_data in robot_data.items():
            gen = int(gen_str)
            if gen not in meeting_rate_per_gen:
                meeting_rate_per_gen[gen] = []
            # Store meeting counts for averaging (divide by 2 since each meeting involves two robots)
            meeting_rate_per_gen[gen].append(gen_data.get("meeting_count", 0) // 2)

    # Calculate average meeting rate per generation
    avg_meeting_rate_per_gen = {}
    for gen, rates in meeting_rate_per_gen.items():
        if rates:
            avg_meeting_rate_per_gen[gen] = np.mean(rates)
        else:
            avg_meeting_rate_per_gen[gen] = None
    return avg_meeting_rate_per_gen


def calculate_offsprings_per_generation(offsprings):
    """Returns a dict: generation -> number of offspring in that generation."""
    offsprings_per_gen = {}
    for robot_id, robot_data in offsprings.items():
        for gen_str, gen_data in robot_data.items():
            gen = int(gen_str)
            if gen not in offsprings_per_gen:
                offsprings_per_gen[gen] = 0
            # Count offspring (divide by 2 since each offspring is counted for both parents)
            offsprings_per_gen[gen] += gen_data.get("offspring_count", 0)

    # Divide by 2 to account for double-counting (each offspring has two parents)
    for gen in offsprings_per_gen:
        offsprings_per_gen[gen] = offsprings_per_gen[gen] // 2

    return offsprings_per_gen


def plot_matings_per_generation(matings_per_gen, config_name, output_dir):
    """Create a plot showing number of matings per generation."""
    generations = sorted(matings_per_gen.keys())
    matings = [matings_per_gen[gen] for gen in generations]

    plt.figure(figsize=(12, 6))
    plt.bar(generations, matings, color="skyblue", alpha=0.7)
    plt.xlabel("Generation")
    plt.ylabel("Number of Matings")
    title = "Number of Matings per Generation"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "matings_per_generation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_meeting_rate_per_generation(meeting_rate_per_gen, config_name, output_dir):
    """Create a plot showing average meeting rate per generation."""
    if not meeting_rate_per_gen:
        print("No meeting rate data available for this experiment")
        return

    generations = sorted(meeting_rate_per_gen.keys())
    meeting_rates = [
        meeting_rate_per_gen[gen]
        for gen in generations
        if meeting_rate_per_gen[gen] is not None
    ]
    meeting_gens = [gen for gen in generations if meeting_rate_per_gen[gen] is not None]

    if not meeting_gens:
        print("No valid meeting rate data available for this experiment")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(
        meeting_gens,
        meeting_rates,
        color="orange",
        marker="o",
        linewidth=2,
        markersize=4,
    )
    plt.xlabel("Generation")
    plt.ylabel("Average Meeting Rate per Robot")
    title = "Average Meeting Rate per Generation"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "meeting_rate_per_generation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_offsprings_per_generation(offsprings_per_gen, config_name, output_dir):
    """Create a plot showing number of offspring per generation."""
    generations = sorted(offsprings_per_gen.keys())
    offsprings = [offsprings_per_gen[gen] for gen in generations]

    plt.figure(figsize=(12, 6))
    plt.bar(generations, offsprings, color="lightgreen", alpha=0.7)
    plt.xlabel("Generation")
    plt.ylabel("Number of Offspring")
    title = "Number of Offspring per Generation"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "offsprings_per_generation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def process_single_file(
    json_file_path, output_dir=None, config_name="", overlay_meeting_rate=True
):
    print(f"\nProcessing {json_file_path}...")
    data = load_data(json_file_path)
    matings_per_gen = calculate_matings_per_generation(data)
    meeting_rate_per_gen = (
        calculate_meeting_rate_per_generation(data) if overlay_meeting_rate else None
    )
    offsprings_per_gen = calculate_offsprings_per_generation(data)

    # Always create the matings plot
    plot_matings_per_generation(matings_per_gen, config_name, output_dir)

    # Always create the offspring plot
    plot_offsprings_per_generation(offsprings_per_gen, config_name, output_dir)

    # Only create meeting rate plot for non-standard setups
    if overlay_meeting_rate:
        plot_meeting_rate_per_generation(meeting_rate_per_gen, config_name, output_dir)

    print(f"Plots saved in '{output_dir}' directory")


def process_all_stats(stats_root, output_root):
    stats_root = Path(stats_root)
    output_root = Path(output_root)
    for dirpath, dirnames, filenames in os.walk(stats_root):
        dirpath = Path(dirpath)
        for filename in filenames:
            if filename == "run1_offsprings_corrected.json":
                rel_dir = dirpath.relative_to(stats_root)
                output_dir = output_root / rel_dir
                json_file_path = dirpath / filename
                config_name = rel_dir.as_posix().replace("/", "_") or rel_dir.name
                overlay_meeting_rate = not is_standard_setup(json_file_path)
                process_single_file(
                    json_file_path, output_dir, config_name, overlay_meeting_rate
                )
    print(
        f"\nAll mating plots have been generated and saved in '{output_root}' mirroring the stats/ structure."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate mating statistics plots from run1_offsprings_corrected.json files in stats/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all-stats
        """,
    )
    parser.add_argument(
        "--all-stats",
        action="store_true",
        help="Process the entire stats folder, creating plots for run1_offsprings_corrected.json in each experiment, outputting to run1_plots_mating/ mirroring the stats/ structure.",
    )
    args = parser.parse_args()
    if args.all_stats:
        stats_dir = Path("stats")
        output_dir = Path("run1_plots_mating")
        if not stats_dir.exists():
            print(f"Error: stats directory '{stats_dir}' does not exist.")
            sys.exit(1)
        output_dir.mkdir(exist_ok=True)
        process_all_stats(stats_dir, output_dir)
        return
    print("Please use --all-stats to process the entire stats folder.")
    sys.exit(1)


if __name__ == "__main__":
    main()
