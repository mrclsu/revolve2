#!/usr/bin/env python3
"""
Script to plot aggregated_final.json files from the stats folder.
Creates both regular and filtered versions of fitness plots.
"""

import json
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
    labels = ["Baseline", "Lowest Fitness", "Max Age"]

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

    # Define colors to match the combined plot
    colors = ["#1f77b4", "#2ca02c", "#d62728"]  # Blue, Green, Red

    # Create the plot
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create violin plot with custom colors
    sns.violinplot(
        data=df,
        y="Experiment",
        x="Fitness",
        ax=ax,
        inner=None,
        alpha=0.25,
        hue="Experiment",
        palette=colors,
        legend=False,
    )

    # Overlay swarm plot with matching colors
    sns.swarmplot(
        data=df,
        y="Experiment",
        x="Fitness",
        ax=ax,
        size=3,
        alpha=0.8,
        hue="Experiment",
        palette=colors,
        legend=False,
    )

    ax.set_xlabel("Fitness (Normalized Path Length)")
    ax.set_ylabel("Experiment")
    ax.set_title("Last Generation Fitness Distribution - NPL-based Fitness Function")
    ax.grid(True, alpha=0.3)

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_age_violin_swarm_plot(aggregated_files, output_path):
    """Create violin and swarm plots for the three NPL-based experiments showing last generation age distribution."""
    cases = ["standard_max_npl", "lowest_fitness_max_npl", "max_age_max_npl"]
    labels = ["Baseline", "Lowest Fitness", "Max Age"]

    # Collect age data for last generation from each case
    age_data = []
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
            age_values = [robot["age"] for robot in robot_stats[last_gen]]

            # Flatten age values if they are lists
            flattened_age = []
            for age in age_values:
                if isinstance(age, list):
                    flattened_age.extend(age)
                else:
                    flattened_age.append(age)

            # Add to data lists
            age_data.extend(flattened_age)
            case_labels.extend([labels[i]] * len(flattened_age))

        except Exception as e:
            print(f"Error processing {case_file}: {e}")
            continue

    # Ensure data is properly structured
    if not age_data or not case_labels:
        print("Error: No age data found for violin plot")
        return

    # Create DataFrame for seaborn
    df = pd.DataFrame({"Age": age_data, "Experiment": case_labels})

    # Define colors to match the combined plot
    colors = ["#1f77b4", "#2ca02c", "#d62728"]  # Blue, Green, Red

    # Create the plot
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create violin plot with custom colors
    sns.violinplot(
        data=df,
        y="Experiment",
        x="Age",
        ax=ax,
        inner=None,
        alpha=0.25,
        hue="Experiment",
        palette=colors,
        legend=False,
    )

    # Overlay swarm plot with matching colors
    sns.swarmplot(
        data=df,
        y="Experiment",
        x="Age",
        ax=ax,
        size=3,
        alpha=0.8,
        hue="Experiment",
        palette=colors,
        legend=False,
    )

    ax.set_xlabel("Age (Generations)")
    ax.set_ylabel("Experiment")
    ax.set_title("Last Generation Age Distribution - NPL-based Fitness Function")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_combined_average_population_plot(
    aggregated_files, output_path, fitness_key: str, runs_per_aggregate: int = 5
):
    """Create a combined plot of average population size per generation for two death mechanisms.

    Plots the average population size (per run) for the two death mechanisms
    (lowest_fitness and max_age) for a given fitness function key
    (one of: "combo", "max_head", "max_npl").
    """
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))

    cases = [
        f"lowest_fitness_{fitness_key}",
        f"max_age_{fitness_key}",
    ]
    labels = ["Lowest Fitness", "Max Age"]
    colors = ["#1f77b4", "#d62728"]  # Blue, Red

    any_series_plotted = False

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
            data = load_aggregated_data(case_file)
            robot_stats = data.get("robot_stats_per_generation", {})
            if not robot_stats:
                continue

            generations = []
            avg_population = []
            for gen in sorted(robot_stats.keys(), key=int):
                generations.append(int(gen))
                avg_population.append(
                    len(robot_stats[gen]) / max(1, runs_per_aggregate)
                )

            if len(generations) == 0:
                continue

            plt.plot(
                generations,
                avg_population,
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

    # Title helper
    title_map = {
        "combo": "Combo",
        "max_head": "Max Head Stability",
        "max_npl": "Max NPL",
    }
    pretty_key = title_map.get(fitness_key, fitness_key)

    plt.xlabel("Generation")
    plt.ylabel("Population Size (per run)")
    plt.title(f"Average Population Size per Generation - {pretty_key}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_average_population_plot(
    data, output_path, title, runs_per_aggregate: int = 5
) -> None:
    """Plot average population size per generation.

    Assumes `data['robot_stats_per_generation']` contains an aggregated list of
    robots across multiple runs for each generation. The average population per
    generation is computed as len(robots) / runs_per_aggregate.
    """
    plt.style.use("seaborn-v0_8")
    plt.figure(figsize=(12, 8))

    robot_stats = data.get("robot_stats_per_generation", {})
    if not robot_stats:
        plt.close()
        return

    generations = []
    avg_population = []

    for gen in sorted(robot_stats.keys(), key=int):
        generations.append(int(gen))
        avg_population.append(len(robot_stats[gen]) / max(1, runs_per_aggregate))

    if len(generations) == 0:
        plt.close()
        return

    plt.plot(generations, avg_population, linewidth=2, label="Average Population Size")

    plt.xlabel("Generation")
    plt.ylabel("Population Size")
    plt.title(f"{title} - Average Population Size (per run)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_experiment_summary(data, output_path, experiment_name):
    """Compute and save simple summary statistics for an experiment to a text file.

    The summary includes:
    - Number of generations and range
    - Population size stats across generations (first, last, min, max, mean)
    - Fitness statistics for the final non-empty generation (count, mean, median, std, min, Q1, Q3, max)
    - Overall best fitness across all generations and the generation it occurred in
    - Mean of per-generation averages and medians
    - Age stats for the final non-empty generation (mean, median, max)
    """
    robot_stats = data.get("robot_stats_per_generation", {})
    if not robot_stats:
        with open(output_path, "w") as f:
            f.write(f"Experiment: {experiment_name}\nNo robot stats available.\n")
        return

    generation_keys = sorted(robot_stats.keys(), key=int)
    num_generations = len(generation_keys)

    # Population sizes per generation
    population_sizes = []
    for gen in generation_keys:
        try:
            population_sizes.append(len(robot_stats[gen]))
        except Exception:
            population_sizes.append(0)

    first_gen = int(generation_keys[0]) if generation_keys else None
    last_gen_any = int(generation_keys[-1]) if generation_keys else None

    # Find the last generation that has any robots
    last_non_empty_gen_key = None
    for gen in reversed(generation_keys):
        if len(robot_stats[gen]) > 0:
            last_non_empty_gen_key = gen
            break

    # Helper to flatten scalar-or-list values
    def _flatten(values):
        flat = []
        for v in values:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        return flat

    # Fitness stats for last non-empty generation
    last_fitness_stats = {}
    last_age_stats = {}
    if last_non_empty_gen_key is not None:
        last_gen_fitness = _flatten(
            [
                r.get("fitness", None)
                for r in robot_stats[last_non_empty_gen_key]
                if r.get("fitness", None) is not None
            ]
        )
        last_gen_age = _flatten(
            [
                r.get("age", None)
                for r in robot_stats[last_non_empty_gen_key]
                if r.get("age", None) is not None
            ]
        )

        if len(last_gen_fitness) > 0:
            arr = np.array(last_gen_fitness, dtype=float)
            last_fitness_stats = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "q1": float(np.percentile(arr, 25)),
                "q3": float(np.percentile(arr, 75)),
                "max": float(np.max(arr)),
            }
        if len(last_gen_age) > 0:
            arr_age = np.array(last_gen_age, dtype=float)
            last_age_stats = {
                "count": int(arr_age.size),
                "mean": float(np.mean(arr_age)),
                "median": float(np.median(arr_age)),
                "max": float(np.max(arr_age)),
            }

    # Overall best fitness across all generations and mean of per-gen stats
    overall_best = None
    overall_best_gen = None
    per_gen_means = []
    per_gen_medians = []
    for gen in generation_keys:
        fitness_vals = _flatten(
            [
                r.get("fitness", None)
                for r in robot_stats[gen]
                if r.get("fitness", None) is not None
            ]
        )
        if len(fitness_vals) == 0:
            continue
        arr = np.array(fitness_vals, dtype=float)
        per_gen_means.append(np.mean(arr))
        per_gen_medians.append(np.median(arr))
        max_here = np.max(arr)
        if overall_best is None or max_here > overall_best:
            overall_best = float(max_here)
            overall_best_gen = int(gen)

    mean_of_means = float(np.mean(per_gen_means)) if per_gen_means else None
    mean_of_medians = float(np.mean(per_gen_medians)) if per_gen_medians else None

    # Prepare text
    lines = []
    lines.append(f"Experiment: {experiment_name}")
    lines.append("")
    lines.append(f"Generations: {num_generations} (range {first_gen}..{last_gen_any})")
    if population_sizes:
        lines.append(
            "Population sizes: first="
            f"{population_sizes[0]}, last={population_sizes[-1]}, min={int(np.min(population_sizes))}, "
            f"max={int(np.max(population_sizes))}, mean={float(np.mean(population_sizes)):.2f}"
        )
    else:
        lines.append("Population sizes: n/a")

    lines.append("")
    lines.append("Fitness (last non-empty generation):")
    if last_fitness_stats:
        lines.append(
            "  count={count}, mean={mean:.6f}, median={median:.6f}, std={std:.6f}, min={min:.6f}, q1={q1:.6f}, q3={q3:.6f}, max={max:.6f}".format(
                **last_fitness_stats
            )
        )
    else:
        lines.append("  n/a")

    lines.append("Age (last non-empty generation):")
    if last_age_stats:
        lines.append(
            "  count={count}, mean={mean:.2f}, median={median:.2f}, max={max:.2f}".format(
                **last_age_stats
            )
        )
    else:
        lines.append("  n/a")

    lines.append("")
    lines.append("Fitness (overall):")
    if overall_best is not None:
        lines.append(f"  best_max={overall_best:.6f} at generation {overall_best_gen}")
    else:
        lines.append("  best_max: n/a")
    lines.append(
        f"  mean_of_means={mean_of_means:.6f}"
        if mean_of_means is not None
        else "  mean_of_means: n/a"
    )
    lines.append(
        f"  mean_of_medians={mean_of_medians:.6f}"
        if mean_of_medians is not None
        else "  mean_of_medians: n/a"
    )

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


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

            # Create average population plot (skip for the three standard_* experiments)
            if experiment_name not in {
                "standard_combo",
                "standard_max_head",
                "standard_max_npl",
            }:
                avg_pop_plot_path = (
                    experiment_output_dir / "average_population_size.png"
                )
                create_average_population_plot(
                    data,
                    avg_pop_plot_path,
                    experiment_name,
                    runs_per_aggregate=5,
                )
                print(f"  Created average population plot: {avg_pop_plot_path}")

            # Save summary statistics
            summary_path = experiment_output_dir / "summary_statistics.txt"
            save_experiment_summary(data, summary_path, experiment_name)
            print(f"  Created summary statistics: {summary_path}")

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue

    # Create combined max_npl plot
    print("\nCreating combined max_npl plot...")
    combined_plot_path = output_dir / "combined_max_npl_cases.png"
    create_combined_max_npl_plot(aggregated_files, combined_plot_path)
    print(f"Created combined plot: {combined_plot_path}")

    # Create violin and swarm plot for last generation fitness
    print("\nCreating violin and swarm plot for last generation fitness...")
    violin_plot_path = output_dir / "violin_swarm_last_generation_npl.png"
    create_violin_swarm_plot(aggregated_files, violin_plot_path)
    print(f"Created fitness violin and swarm plot: {violin_plot_path}")

    # Create violin and swarm plot for last generation age
    print("\nCreating violin and swarm plot for last generation age...")
    age_violin_plot_path = output_dir / "violin_swarm_age_last_generation_npl.png"
    create_age_violin_swarm_plot(aggregated_files, age_violin_plot_path)
    print("Created age violin and swarm plot:", age_violin_plot_path)

    # Create combined average population plots for each fitness function
    print(
        "\nCreating combined average population plots (per run, 5 runs aggregated)..."
    )
    avg_pop_combo_path = output_dir / "average_population_combo.png"
    avg_pop_max_head_path = output_dir / "average_population_max_head.png"
    avg_pop_max_npl_path = output_dir / "average_population_max_npl.png"

    create_combined_average_population_plot(
        aggregated_files, avg_pop_combo_path, fitness_key="combo", runs_per_aggregate=5
    )
    print("Created combined average population plot:", avg_pop_combo_path)

    create_combined_average_population_plot(
        aggregated_files,
        avg_pop_max_head_path,
        fitness_key="max_head",
        runs_per_aggregate=5,
    )
    print("Created combined average population plot:", avg_pop_max_head_path)

    create_combined_average_population_plot(
        aggregated_files,
        avg_pop_max_npl_path,
        fitness_key="max_npl",
        runs_per_aggregate=5,
    )
    print("Created combined average population plot:", avg_pop_max_npl_path)

    print("\nAll plots saved to:", output_dir.absolute())


if __name__ == "__main__":
    main()
