#!/usr/bin/env python3
"""
Script to plot aggregated_final.json files from the stats folder.
Creates both regular and filtered versions of fitness plots.
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import stats
import seaborn as sns


def load_aggregated_data(file_path):
    """Load aggregated data from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def extract_fitness_stats(data):
    """Extract fitness statistics for each generation."""
    generations = []
    avg_fitness = []
    std_fitness = []
    median_fitness = []
    min_fitness = []
    max_fitness = []

    robot_stats = data["robot_stats_per_generation"]

    for gen in sorted(robot_stats.keys(), key=int):
        fitness_values = [robot["fitness"] for robot in robot_stats[gen]]

        generations.append(int(gen))
        avg_fitness.append(np.mean(fitness_values))
        std_fitness.append(np.std(fitness_values))
        median_fitness.append(np.median(fitness_values))
        min_fitness.append(np.min(fitness_values))
        max_fitness.append(np.max(fitness_values))

    return {
        "generations": generations,
        "avg_fitness": avg_fitness,
        "std_fitness": std_fitness,
        "median_fitness": median_fitness,
        "min_fitness": min_fitness,
        "max_fitness": max_fitness,
    }


def extract_filtered_fitness_stats(data, outlier_threshold=2.0):
    """Extract fitness statistics with outlier filtering."""
    generations = []
    avg_fitness = []
    std_fitness = []
    median_fitness = []
    min_fitness = []
    max_fitness = []

    robot_stats = data["robot_stats_per_generation"]

    for gen in sorted(robot_stats.keys(), key=int):
        fitness_values = np.array([robot["fitness"] for robot in robot_stats[gen]])

        # Remove perfect scores
        z_scores = np.abs(stats.zscore(fitness_values))
        filtered_fitness = fitness_values[z_scores < outlier_threshold]

        if len(filtered_fitness) > 0:
            generations.append(int(gen))
            avg_fitness.append(np.mean(filtered_fitness))
            std_fitness.append(np.std(filtered_fitness))
            median_fitness.append(np.median(filtered_fitness))
            min_fitness.append(np.min(filtered_fitness))
            max_fitness.append(np.max(filtered_fitness))

    return {
        "generations": generations,
        "avg_fitness": avg_fitness,
        "std_fitness": std_fitness,
        "median_fitness": median_fitness,
        "min_fitness": min_fitness,
        "max_fitness": max_fitness,
    }


def create_fitness_plot(stats_data, output_path, title, filtered=False):
    """Create a comprehensive fitness plot."""
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))

    generations = stats_data["generations"]

    # Plot mean with standard deviation
    plt.fill_between(
        generations,
        np.array(stats_data["avg_fitness"]) - np.array(stats_data["std_fitness"]),
        np.array(stats_data["avg_fitness"]) + np.array(stats_data["std_fitness"]),
        alpha=0.3,
        label="±1 Std Dev",
    )

    # Plot statistics
    plt.plot(
        generations,
        stats_data["avg_fitness"],
        "b-",
        linewidth=2,
        label="Average Fitness",
    )
    plt.plot(
        generations,
        stats_data["median_fitness"],
        "g-",
        linewidth=2,
        label="Median Fitness",
    )
    plt.plot(
        generations, stats_data["min_fitness"], "r--", linewidth=1, label="Min Fitness"
    )
    plt.plot(
        generations, stats_data["max_fitness"], "r-", linewidth=1, label="Max Fitness"
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"{title} - Fitness Statistics Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_median_fitness_plot(
    stats_data, output_path, title, filtered=False, raw_data=None
):
    """Create a plot showing median fitness with first and third quantiles as bounds."""
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))

    generations = stats_data["generations"]
    avg_fitness = stats_data["avg_fitness"]

    # Calculate first and third quantiles for each generation
    if raw_data is not None:
        robot_stats = raw_data["robot_stats_per_generation"]
        q1_values = []
        q3_values = []

        for gen in sorted(robot_stats.keys(), key=int):
            fitness_values = [robot["fitness"] for robot in robot_stats[gen]]
            q1_values.append(np.percentile(fitness_values, 25))
            q3_values.append(np.percentile(fitness_values, 75))
    else:
        # Fallback: use constant bounds based on overall median distribution
        q1 = np.percentile(avg_fitness, 25)
        q3 = np.percentile(avg_fitness, 75)
        q1_values = [q1] * len(generations)
        q3_values = [q3] * len(generations)

    # Plot median line
    plt.plot(
        generations,
        avg_fitness,
        "b-",
        linewidth=3,
        label="Average Fitness",
    )

    # Fill area between Q1 and Q3 as moving bounds
    plt.fill_between(generations, q1_values, q3_values, alpha=0.3, label="Q1-Q3 Range")

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(f"{title} - Average Fitness with Quantile Bounds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_combined_max_npl_plot(aggregated_files, output_path):
    """Create a combined plot of the three max_npl cases on a single plot."""
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(14, 10))

    cases = ["standard_max_npl", "lowest_fitness_max_npl", "max_age_max_npl"]
    labels = ["Baseline", "Lowest Fitness Death Mechanism", "Max Age Death Mechanism"]

    for i, case in enumerate(cases):
        # Find the corresponding file
        case_file = None
        for file_path in aggregated_files:
            if case in file_path:
                case_file = file_path
                break

        if case_file is None:
            print(f"Warning: Could not find file for case {case}")
            continue

        try:
            # Load data
            data = load_aggregated_data(case_file)
            stats_data = extract_fitness_stats(data)

            generations = stats_data["generations"]
            avg_fitness = stats_data["avg_fitness"]

            # Calculate Q1 and Q3 for each generation
            robot_stats = data["robot_stats_per_generation"]
            q1_values = []
            q3_values = []

            for gen in sorted(robot_stats.keys(), key=int):
                fitness_values = [robot["fitness"] for robot in robot_stats[gen]]
                q1_values.append(np.percentile(fitness_values, 25))
                q3_values.append(np.percentile(fitness_values, 75))

            # Plot average line
            plt.plot(
                generations,
                avg_fitness,
                linewidth=2,
                label=labels[i],
            )

            # Fill area between Q1 and Q3
            plt.fill_between(generations, q1_values, q3_values, alpha=0.2)

        except Exception as e:
            print(f"Error processing {case_file}: {e}")
            continue

    plt.xlabel("Generation")
    plt.ylabel("Fitness (Normalized Path Length)")
    plt.title("Average Fitness with 1st and 3rd Quantile Bounds")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_violin_swarm_plot(aggregated_files, output_path):
    """Create violin and swarm plots for the three NPL-based experiments showing last generation fitness."""
    cases = ["standard_max_npl", "lowest_fitness_max_npl", "max_age_max_npl"]
    labels = ["Baseline", "Lowest Fitness Death Mechanism", "Max Age Death Mechanism"]

    # Collect fitness data for last generation from each case
    fitness_data = []
    case_labels = []

    for i, case in enumerate(cases):
        # Find the corresponding file
        case_file = None
        for file_path in aggregated_files:
            if case in file_path:
                case_file = file_path
                break

        if case_file is None:
            print(f"Warning: Could not find file for case {case}")
            continue

        try:
            # Load data
            data = load_aggregated_data(case_file)
            robot_stats = data["robot_stats_per_generation"]

            # Get last generation
            last_gen = max(robot_stats.keys(), key=int)
            fitness_values = [robot["fitness"] for robot in robot_stats[last_gen]]

            # Flatten fitness values if they are lists
            flattened_fitness = []
            for fitness in fitness_values:
                if isinstance(fitness, list):
                    flattened_fitness.extend(fitness)
                else:
                    flattened_fitness.append(fitness)

            # Add to data lists
            fitness_data.extend(flattened_fitness)
            case_labels.extend([labels[i]] * len(flattened_fitness))

        except Exception as e:
            print(f"Error processing {case_file}: {e}")
            continue

    # Ensure data is properly structured
    if not fitness_data or not case_labels:
        print("Error: No data found for violin plot")
        return

    # Create DataFrame for seaborn
    df = pd.DataFrame({"Fitness": fitness_data, "Experiment": case_labels})

    # Create the plot
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create violin plot
    sns.violinplot(data=df, y="Experiment", x="Fitness", ax=ax, inner=None, alpha=0.7)

    # Overlay swarm plot
    sns.swarmplot(data=df, y="Experiment", x="Fitness", ax=ax, size=3, alpha=0.6)

    ax.set_xlabel("Fitness (Normalized Path Length)")
    ax.set_ylabel("Experiment")
    ax.set_title("Last Generation Fitness Distribution - NPL-based Experiments")
    ax.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def find_aggregated_files():
    """Find all aggregated JSON files in the aggregated_stats folder."""
    files = glob.glob("aggregated_stats/*_aggregated.json")
    return files


def main():
    """Main function to process all aggregated files and create plots."""
    # Create output directory
    output_dir = Path("aggregated_plots")
    output_dir.mkdir(exist_ok=True)

    # Find all aggregated files
    aggregated_files = find_aggregated_files()

    print(f"Found {len(aggregated_files)} aggregated JSON files:")
    for file_path in aggregated_files:
        print(f"  - {file_path}")

    # Process each file
    for file_path in aggregated_files:
        print(f"\nProcessing {file_path}...")

        # Extract experiment name from filename
        file_path_obj = Path(file_path)
        experiment_name = file_path_obj.stem.replace("_aggregated", "")
        print(f"Experiment: {experiment_name}")

        # Create experiment output directory
        experiment_output_dir = output_dir / experiment_name
        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load data
            data = load_aggregated_data(file_path)

            # Extract regular statistics
            stats_data = extract_fitness_stats(data)

            # Create regular plot
            regular_plot_path = experiment_output_dir / "fitness_statistics.png"
            create_fitness_plot(
                stats_data, regular_plot_path, experiment_name, filtered=False
            )
            print(f"  Created regular plot: {regular_plot_path}")

            # Extract filtered statistics
            filtered_stats_data = extract_filtered_fitness_stats(data)

            # Create filtered plot
            filtered_plot_path = (
                experiment_output_dir / "fitness_statistics_filtered.png"
            )
            create_fitness_plot(
                filtered_stats_data,
                filtered_plot_path,
                f"{experiment_name} (Filtered)",
                filtered=True,
            )
            print(f"  Created filtered plot: {filtered_plot_path}")

            # Create median fitness plot (regular)
            median_fitness_plot_path = (
                experiment_output_dir / "median_fitness_with_quantiles.png"
            )
            create_median_fitness_plot(
                stats_data,
                median_fitness_plot_path,
                f"{experiment_name} (Median with Quantiles)",
                filtered=False,
                raw_data=data,
            )
            print(f"  Created median fitness plot: {median_fitness_plot_path}")

            # Create median fitness plot (filtered)
            filtered_median_fitness_plot_path = (
                experiment_output_dir / "median_fitness_with_quantiles_filtered.png"
            )
            create_median_fitness_plot(
                filtered_stats_data,
                filtered_median_fitness_plot_path,
                f"{experiment_name} (Median with Quantiles, Filtered)",
                filtered=True,
                raw_data=data,
            )
            print(
                f"  Created filtered median fitness plot: {filtered_median_fitness_plot_path}"
            )

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue

    # Create combined max_npl plot
    print(f"\nCreating combined max_npl plot...")
    combined_plot_path = output_dir / "combined_max_npl_cases.png"
    create_combined_max_npl_plot(aggregated_files, combined_plot_path)
    print(f"Created combined plot: {combined_plot_path}")

    # Create violin and swarm plot for last generation
    print(f"\nCreating violin and swarm plot for last generation...")
    violin_plot_path = output_dir / "violin_swarm_last_generation_npl.png"
    create_violin_swarm_plot(aggregated_files, violin_plot_path)
    print(f"Created violin and swarm plot: {violin_plot_path}")

    print(f"\nAll plots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
