#!/usr/bin/env python3
"""
Runtime Analysis Script

This script analyzes runtime data from a JSONL file and generates plots
showing performance metrics (max, average, standard deviation) for different
robot types and counts.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def load_data(file_path: str) -> Dict[str, Dict[int, List[float]]]:
    """
    Load runtime data from JSONL file.

    Args:
        file_path: Path to the JSONL file

    Returns:
        Dictionary with robot types as keys and robot counts as nested keys
    """
    data = {}

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            robot_type = entry["robot_type"]
            robot_count = entry["robot_count"]
            results = entry["results"]

            if robot_type not in data:
                data[robot_type] = {}

            data[robot_type][robot_count] = results

    return data


def calculate_statistics(
    data: Dict[str, Dict[int, List[float]]],
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Calculate statistics for each robot type and count.

    Args:
        data: Loaded data dictionary

    Returns:
        Dictionary with calculated statistics
    """
    stats = {}

    for robot_type, robot_counts in data.items():
        stats[robot_type] = {}
        for robot_count, results in robot_counts.items():
            results_array = np.array(results)
            stats[robot_type][robot_count] = {
                "max": np.max(results_array),
                "min": np.min(results_array),
                "median": np.median(results_array),
                "average": np.mean(results_array),
                "std": np.std(results_array),
            }

    return stats


def create_per_robot_plot(
    stats: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str = "runtime_plots",
    label: str = "",
):
    """
    Create a plot showing average runtime per robot for each robot type.

    Args:
        stats: Calculated statistics
        output_dir: Directory to save plots
    """
    # Robot counts to plot (1-100)
    robot_counts = sorted(list(next(iter(stats.values())).keys()))

    # Create single figure for all robot types
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        f"Average Runtime per Robot for 5 simulation seconds - {label}", fontsize=16
    )

    # Plot each robot type
    for robot_type in stats.keys():
        avg_values = [stats[robot_type][count]["average"] for count in robot_counts]
        std_values = [stats[robot_type][count]["std"] for count in robot_counts]
        per_robot_values = [avg / count for avg, count in zip(avg_values, robot_counts)]
        per_robot_std_values = [
            std / count for std, count in zip(std_values, robot_counts)
        ]

        # Plot average line
        ax.plot(
            robot_counts,
            per_robot_values,
            linewidth=2,
            label=f"{robot_type.capitalize()} Robots",
            marker="o",
            markersize=4,
        )

        # Add shading for standard deviation
        upper_bound = [
            avg + std for avg, std in zip(per_robot_values, per_robot_std_values)
        ]
        lower_bound = [
            avg - std for avg, std in zip(per_robot_values, per_robot_std_values)
        ]

        ax.fill_between(
            robot_counts,
            lower_bound,
            upper_bound,
            alpha=0.2,
            label=f"{robot_type.capitalize()} Std Dev",
        )

    # Customize the plot
    ax.set_xlabel("Robot Count")
    ax.set_ylabel("Average Runtime per Robot (seconds)")
    ax.set_title("Runtime Efficiency vs Robot Count")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set y-axis to start from 0 for better visualization
    ax.set_ylim(bottom=0)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/per_robot_runtime_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Generated per-robot plot: {output_dir}/per_robot_runtime_analysis.png")


def create_combined_plot(
    stats: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str = "runtime_plots",
    label: str = "",
):
    """
    Create a combined plot showing min, max, avg, and std for both random and spider robots on one chart.
    """
    robot_counts = sorted(list(next(iter(stats.values())).keys()))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(
        f"Runtime Performance of 5 simulation seconds - Combined Robots - {label}",
        fontsize=16,
    )

    for robot_type in stats.keys():
        avg_values = [stats[robot_type][count]["average"] for count in robot_counts]
        std_values = [stats[robot_type][count]["std"] for count in robot_counts]

        # Plot average line with shading for standard deviation
        ax.plot(
            robot_counts,
            avg_values,
            linewidth=3,
            label=f"{robot_type.capitalize()} Average",
            zorder=3,
            marker="o",
        )
        upper_bound = [avg + std for avg, std in zip(avg_values, std_values)]
        lower_bound = [avg - std for avg, std in zip(avg_values, std_values)]
        ax.fill_between(
            robot_counts,
            lower_bound,
            upper_bound,
            alpha=0.15,
            label=f"{robot_type.capitalize()} Std Dev",
        )

    ax.set_xlabel("Robot Count")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Runtime Statistics vs Robot Count (Combined)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/combined_runtime_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Generated combined plot: {output_dir}/combined_runtime_analysis.png")


def create_plots(
    stats: Dict[str, Dict[int, Dict[str, float]]],
    output_dir: str = "runtime_plots",
    label: str = "",
):
    """
    Create plots for each robot type showing max, min, average, and ±1 std shading.
    Also creates a combined plot for all robot types.
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8")

    # Robot counts to plot (1-100)
    robot_counts = sorted(list(next(iter(stats.values())).keys()))

    # Calculate global y-axis limits for consistent scaling
    all_max_values = []
    all_min_values = []
    for robot_type in stats.keys():
        for count in robot_counts:
            all_max_values.append(stats[robot_type][count]["max"])
            all_min_values.append(stats[robot_type][count]["min"])

    global_max = max(all_max_values)
    global_min = min(all_min_values)
    y_margin = (global_max - global_min) * 0.05  # 5% margin

    for robot_type in stats.keys():
        # Extract data for this robot type
        max_values = [stats[robot_type][count]["max"] for count in robot_counts]
        min_values = [stats[robot_type][count]["min"] for count in robot_counts]
        avg_values = [stats[robot_type][count]["average"] for count in robot_counts]
        std_values = [stats[robot_type][count]["std"] for count in robot_counts]

        # Create single figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(
            f"Runtime Performance of 5 simulation seconds - {robot_type.capitalize()} Robots - {label}",
            fontsize=16,
        )

        # Plot average line with shading for standard deviation
        ax.plot(
            robot_counts,
            avg_values,
            linewidth=3,
            label="Average",
            zorder=3,
            marker="o",
        )

        # Add shading for standard deviation (average ± std)
        upper_bound = [avg + std for avg, std in zip(avg_values, std_values)]
        lower_bound = [avg - std for avg, std in zip(avg_values, std_values)]

        ax.fill_between(
            robot_counts,
            lower_bound,
            upper_bound,
            alpha=0.2,
            label="Standard Deviation",
        )

        # Plot maximum values
        ax.plot(
            robot_counts,
            max_values,
            linewidth=2,
            label="Maximum",
            zorder=2,
            marker="o",
        )

        # Plot minimum values
        ax.plot(
            robot_counts, min_values, linewidth=2, label="Minimum", zorder=1, marker="o"
        )

        # Customize the plot
        ax.set_xlabel("Robot Count")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_title(f"Runtime Statistics vs Robot Count")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set consistent y-axis limits across all plots
        ax.set_ylim(global_min - y_margin, global_max + y_margin)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{robot_type}_runtime_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(
            f"Generated plot for {robot_type} robots: {output_dir}/{robot_type}_runtime_analysis.png"
        )

    # Create the per-robot plot
    create_per_robot_plot(stats, output_dir, label)
    # Create the combined plot
    create_combined_plot(stats, output_dir, label)


def print_summary(
    stats: Dict[str, Dict[int, Dict[str, float]]], output_dir: str = "runtime_plots"
):
    """
    Print a summary of the statistics and save to a results.txt file.

    Args:
        stats: Calculated statistics
        output_dir: Directory to save results.txt
    """
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("RUNTIME ANALYSIS SUMMARY")
    lines.append("=" * 60)

    for robot_type in stats.keys():
        lines.append(f"\n{robot_type.upper()} ROBOTS:")
        lines.append("-" * 40)

        # Find the robot count with maximum average runtime
        max_avg_count = max(
            stats[robot_type].keys(), key=lambda x: stats[robot_type][x]["average"]
        )
        max_avg_time = stats[robot_type][max_avg_count]["average"]

        # Find the robot count with maximum standard deviation
        max_std_count = max(
            stats[robot_type].keys(), key=lambda x: stats[robot_type][x]["std"]
        )
        max_std_time = stats[robot_type][max_std_count]["std"]

        lines.append(
            f"Robot count with highest average runtime: {max_avg_count} robots ({max_avg_time:.3f}s)"
        )
        lines.append(
            f"Robot count with highest variability: {max_std_count} robots (std: {max_std_time:.3f}s)"
        )

        # Show statistics for all robot counts found in the file
        all_counts = sorted(stats[robot_type].keys())
        lines.append(f"\nAll statistics ({len(all_counts)} robot counts):")
        for count in all_counts:
            stat = stats[robot_type][count]
            lines.append(
                f"  {count:3d} robots: avg={stat['average']:6.3f}s, min={stat['min']:6.3f}s, max={stat['max']:6.3f}s, std={stat['std']:6.3f}s, median={stat['median']:6.3f}s"
            )

    summary = "\n".join(lines)
    print(summary)

    # Write to results.txt in the output directory
    results_path = Path(output_dir) / "results.txt"
    with open(results_path, "w") as f:
        f.write(summary)
    print(f"\nSummary written to {results_path}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze runtime data from JSONL file")
    parser.add_argument("input_file", help="Path to the JSONL input file")
    parser.add_argument(
        "--output-dir",
        default="runtime_plots",
        help="Output directory for plots (default: runtime_plots)",
    )

    parser.add_argument(
        "--label",
        default="",
        help="Label to add to the main plot title",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    try:
        # Load and analyze data
        print(f"Loading data from {args.input_file}...")
        data = load_data(args.input_file)

        if not data:
            print("Error: No data found in the input file.")
            return

        print(f"Found data for robot types: {list(data.keys())}")

        # Calculate statistics
        print("Calculating statistics...")
        stats = calculate_statistics(data)

        # Create plots
        print("Generating plots...")
        create_plots(stats, args.output_dir, args.label)

        # Print summary
        print_summary(stats, args.output_dir)

        print(f"\nAnalysis complete! Plots saved in '{args.output_dir}' directory.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        return


if __name__ == "__main__":
    main()
