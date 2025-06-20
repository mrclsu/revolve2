#!/usr/bin/env python3
"""
Script to plot evolution statistics from JSON data.
Creates three plots:
1. Population size over generations
2. Population size with births/deaths overlay
3. Average age of robots across generations

Usage:
    python plot_evolution_stats.py <file_or_folder_path>

Examples:
    python plot_evolution_stats.py saved_stats/very_first_implementation.json
    python plot_evolution_stats.py saved_stats/early_results_6_confs
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def load_data(filename):
    """Load the JSON data from file."""
    with open(filename, "r") as f:
        return json.load(f)


def calculate_births_deaths(robot_stats):
    """Calculate births and deaths per generation."""
    births = defaultdict(int)
    deaths = defaultdict(int)

    for robot_id, stats in robot_stats.items():
        initial_gen = stats["initial_generation"]
        final_gen = stats["final_generation"]

        births[initial_gen] += 1

        if final_gen is not None:
            deaths[final_gen] += 1

    return births, deaths


def calculate_age_statistics_per_generation(robot_stats, max_generation):
    """Calculate the average and median age of robots alive in each generation."""
    age_per_generation = defaultdict(list)

    for robot_id, stats in robot_stats.items():
        initial_gen = stats["initial_generation"]
        final_gen = stats["final_generation"]

        # If robot is still alive, consider it alive until max_generation
        if final_gen is None:
            final_gen = max_generation

        # For each generation the robot was alive, calculate its age
        for gen in range(initial_gen, final_gen + 1):
            age = gen - initial_gen
            age_per_generation[gen].append(age)

    # Calculate average, median, and maximum age per generation
    avg_ages = {}
    median_ages = {}
    max_ages = {}
    for gen in range(max_generation + 1):
        if age_per_generation[gen]:
            avg_ages[gen] = np.mean(age_per_generation[gen])
            median_ages[gen] = np.median(age_per_generation[gen])
            max_ages[gen] = np.max(age_per_generation[gen])
        else:
            avg_ages[gen] = 0  # No robots alive in this generation
            median_ages[gen] = 0
            max_ages[gen] = 0

    return avg_ages, median_ages, max_ages


def calculate_fitness_statistics_per_generation(robot_stats, max_generation):
    """Calculate the average, median, and maximum fitness per generation."""
    fitness_per_generation = defaultdict(list)

    for robot_id, stats in robot_stats.items():
        # Check if this robot has fitness data
        if "fitness" in stats and isinstance(stats["fitness"], dict):
            # Each generation this robot was evaluated, add its fitness
            for gen_str, fitness_value in stats["fitness"].items():
                gen = int(gen_str)
                if fitness_value is not None:  # Skip None fitness values
                    fitness_per_generation[gen].append(fitness_value)

    # Calculate average, median, and maximum fitness per generation
    avg_fitness = {}
    median_fitness = {}
    max_fitness = {}
    min_fitness = {}

    for gen in range(max_generation + 1):
        if fitness_per_generation[gen]:
            avg_fitness[gen] = np.mean(fitness_per_generation[gen])
            median_fitness[gen] = np.median(fitness_per_generation[gen])
            max_fitness[gen] = np.max(fitness_per_generation[gen])
            min_fitness[gen] = np.min(fitness_per_generation[gen])
        else:
            # No fitness data for this generation
            avg_fitness[gen] = None
            median_fitness[gen] = None
            max_fitness[gen] = None
            min_fitness[gen] = None

    return avg_fitness, median_fitness, max_fitness, min_fitness


def plot_population_size(generation_to_population_size, config_name=""):
    """Create a plot showing population size over generations."""
    generations = [int(gen) for gen in generation_to_population_size.keys()]
    population_sizes = [generation_to_population_size[str(gen)] for gen in generations]

    plt.figure(figsize=(12, 6))
    plt.plot(generations, population_sizes, "b-", linewidth=2, marker="o", markersize=3)
    plt.xlabel("Generation")
    plt.ylabel("Population Size")
    title = f"Population Size Over Generations"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_population_with_births_deaths(
    generation_to_population_size, robot_stats, config_name=""
):
    """Create a plot showing population size with births/deaths overlay."""
    generations = [int(gen) for gen in generation_to_population_size.keys()]
    population_sizes = [generation_to_population_size[str(gen)] for gen in generations]

    births, deaths = calculate_births_deaths(robot_stats)

    # Create arrays for births and deaths aligned with generations
    births_array = [births[gen] for gen in generations]
    deaths_array = [deaths[gen] for gen in generations]

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot population size as a line
    color = "tab:blue"
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Population Size", color=color)
    line = ax1.plot(
        generations,
        population_sizes,
        color=color,
        linewidth=2,
        marker="o",
        markersize=3,
        label="Population Size",
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for births/deaths
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Births/Deaths Count", color=color)

    # Plot births and deaths as bar charts
    width = 2.5
    x_pos = np.array(generations)
    bars1 = ax2.bar(
        x_pos - width / 2, births_array, width, label="Births", color="green", alpha=0.7
    )
    bars2 = ax2.bar(
        x_pos + width / 2, deaths_array, width, label="Deaths", color="red", alpha=0.7
    )

    ax2.tick_params(axis="y", labelcolor=color)

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    title = "Population Size with Births and Deaths Over Generations"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_average_age(generation_to_population_size, robot_stats, config_name=""):
    """Create a plot showing average, median, and maximum age of robots across generations."""
    generations = [int(gen) for gen in generation_to_population_size.keys()]
    max_generation = max(generations)

    avg_ages, median_ages, max_ages = calculate_age_statistics_per_generation(
        robot_stats, max_generation
    )

    # Extract average, median, and maximum ages for each generation
    avg_age_values = [avg_ages[gen] for gen in generations]
    median_age_values = [median_ages[gen] for gen in generations]
    max_age_values = [max_ages[gen] for gen in generations]

    plt.figure(figsize=(12, 6))
    plt.plot(
        generations,
        avg_age_values,
        "g-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Mean Age",
    )
    plt.plot(
        generations,
        median_age_values,
        "orange",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Median Age",
    )
    plt.plot(
        generations,
        max_age_values,
        "red",
        linewidth=2,
        marker="^",
        markersize=4,
        label="Oldest Age",
    )
    plt.xlabel("Generation")
    plt.ylabel("Age (generations)")
    title = "Mean, Median, and Oldest Age of Robots Across Generations"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_births_deaths_only(generation_to_population_size, robot_stats, config_name=""):
    """Create a plot showing only births and deaths per generation."""
    generations = sorted([int(gen) for gen in generation_to_population_size.keys()])

    births, deaths = calculate_births_deaths(robot_stats)

    births_array = [births.get(gen, 0) for gen in generations]
    deaths_array = [deaths.get(gen, 0) for gen in generations]

    plt.figure(figsize=(14, 7))

    x_pos = np.array(generations)
    width = 0.4

    plt.bar(
        x_pos - width / 2, births_array, width, label="Births", color="green", alpha=0.7
    )
    plt.bar(
        x_pos + width / 2, deaths_array, width, label="Deaths", color="red", alpha=0.7
    )

    plt.xlabel("Generation")
    plt.ylabel("Count")
    title = "Births and Deaths per Generation"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(
        ticks=x_pos[::5], labels=x_pos[::5]
    )  # Show ticks every 5 generations for clarity
    plt.tight_layout()
    return plt.gcf()


def plot_fitness_statistics(generation_to_population_size, robot_stats, config_name=""):
    """Create a plot showing average, median, and maximum fitness across generations."""
    generations = [int(gen) for gen in generation_to_population_size.keys()]
    max_generation = max(generations)

    avg_fitness, median_fitness, max_fitness, min_fitness = (
        calculate_fitness_statistics_per_generation(robot_stats, max_generation)
    )

    # Extract fitness values for each generation, filtering out None values
    generations_with_data = []
    avg_fitness_values = []
    median_fitness_values = []
    max_fitness_values = []
    min_fitness_values = []

    for gen in generations:
        if avg_fitness[gen] is not None:
            generations_with_data.append(gen)
            avg_fitness_values.append(avg_fitness[gen])
            median_fitness_values.append(median_fitness[gen])
            max_fitness_values.append(max_fitness[gen])
            min_fitness_values.append(min_fitness[gen])

    if not generations_with_data:
        print("Warning: No fitness data found in robot_stats")
        return None

    plt.figure(figsize=(12, 6))
    plt.plot(
        generations_with_data,
        avg_fitness_values,
        "b-",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Mean Fitness",
    )
    plt.plot(
        generations_with_data,
        median_fitness_values,
        "orange",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Median Fitness",
    )
    plt.plot(
        generations_with_data,
        max_fitness_values,
        "g-",
        linewidth=2,
        marker="^",
        markersize=4,
        label="Max Fitness",
    )
    plt.plot(
        generations_with_data,
        min_fitness_values,
        "r-",
        linewidth=2,
        marker="v",
        markersize=4,
        label="Min Fitness",
    )
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    title = "Fitness Statistics Across Generations"
    if config_name:
        title += f" - {config_name}"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def print_summary_stats(generation_to_population_size, robot_stats, config_name=""):
    """Print summary statistics."""
    generations = [int(gen) for gen in generation_to_population_size.keys()]
    population_sizes = [generation_to_population_size[str(gen)] for gen in generations]

    births, deaths = calculate_births_deaths(robot_stats)
    avg_ages, median_ages, max_ages = calculate_age_statistics_per_generation(
        robot_stats, max(generations)
    )

    # Calculate fitness statistics
    avg_fitness, median_fitness, max_fitness, min_fitness = (
        calculate_fitness_statistics_per_generation(robot_stats, max(generations))
    )

    header = "=== Evolution Summary Statistics ==="
    if config_name:
        header = f"=== Evolution Summary Statistics - {config_name} ==="
    print(header)
    print(f"Total generations: {max(generations) + 1}")
    print(f"Initial population: {population_sizes[0]}")
    print(f"Final population: {population_sizes[-1]}")
    print(f"Max population: {max(population_sizes)}")
    print(f"Min population: {min(population_sizes)}")
    print(f"Total robots created: {len(robot_stats)}")
    print(f"Total births: {sum(births.values())}")
    print(f"Total deaths: {sum(deaths.values())}")
    print(
        f"Robots still alive: {len([r for r in robot_stats.values() if r['final_generation'] is None])}"
    )
    print(
        f"Average age in final generation: {avg_ages[max(generations)]:.2f} generations"
    )
    print(
        f"Median age in final generation: {median_ages[max(generations)]:.2f} generations"
    )
    print(f"Maximum average age reached: {max(avg_ages.values()):.2f} generations")
    print(f"Maximum median age reached: {max(median_ages.values()):.2f} generations")
    print(f"Maximum oldest age reached: {max(max_ages.values()):.2f} generations")

    # Print fitness statistics if available
    fitness_values = [f for f in avg_fitness.values() if f is not None]
    if fitness_values:
        final_gen_fitness = avg_fitness[max(generations)]
        if final_gen_fitness is not None:
            print(f"Average fitness in final generation: {final_gen_fitness:.4f}")

        max_avg_fitness = max(fitness_values)
        max_fitness_values = [f for f in max_fitness.values() if f is not None]
        if max_fitness_values:
            print(f"Best fitness ever achieved: {max(max_fitness_values):.4f}")
            print(f"Best average fitness: {max_avg_fitness:.4f}")
    else:
        print("No fitness data available in robot_stats")


def process_single_file(json_file_path, output_dir=None, config_name=""):
    """Process a single JSON file and generate all plots."""
    print(f"\nProcessing {json_file_path}...")

    # Load data
    data = load_data(json_file_path)
    generation_to_population_size = data["generation_to_population_size"]
    robot_stats = data["robot_stats"]

    # Print summary statistics
    print_summary_stats(generation_to_population_size, robot_stats, config_name)

    # Set up output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create and save plots
    print("Creating population size plot...")
    fig1 = plot_population_size(generation_to_population_size, config_name)
    output_path = (
        os.path.join(output_dir, "population_size_over_time.png")
        if output_dir
        else "population_size_over_time.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Creating population with births/deaths plot...")
    fig2 = plot_population_with_births_deaths(
        generation_to_population_size, robot_stats, config_name
    )
    output_path = (
        os.path.join(output_dir, "population_with_births_deaths.png")
        if output_dir
        else "population_with_births_deaths.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Creating average age plot...")
    fig3 = plot_average_age(generation_to_population_size, robot_stats, config_name)
    output_path = (
        os.path.join(output_dir, "average_age_over_time.png")
        if output_dir
        else "average_age_over_time.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Creating births and deaths plot...")
    fig4 = plot_births_deaths_only(
        generation_to_population_size, robot_stats, config_name
    )
    output_path = (
        os.path.join(output_dir, "births_deaths_per_generation.png")
        if output_dir
        else "births_deaths_per_generation.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Creating fitness statistics plot...")
    fig5 = plot_fitness_statistics(
        generation_to_population_size, robot_stats, config_name
    )
    if fig5 is not None:
        output_path = (
            os.path.join(output_dir, "fitness_statistics_over_time.png")
            if output_dir
            else "fitness_statistics_over_time.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("Skipped fitness plot due to missing fitness data")

    if output_dir:
        print(f"Plots saved in '{output_dir}' directory")
    else:
        print("Plots saved in current directory")


def process_folder(folder_path):
    """Process all JSON files in a folder."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return

    # Find all JSON files in the folder
    json_files = list(folder_path.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in '{folder_path}'")
        return

    print(f"Found {len(json_files)} JSON files in '{folder_path}'")

    # Create main output directory
    output_base_dir = folder_path / "plots"
    output_base_dir.mkdir(exist_ok=True)

    # Process each JSON file
    for json_file in sorted(json_files):
        # Create config name from filename (remove .json extension)
        config_name = json_file.stem

        # Create subdirectory for this config
        config_output_dir = output_base_dir / config_name

        # Process the file
        process_single_file(json_file, config_output_dir, config_name)

    print(f"\nAll plots have been generated and saved in '{output_base_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evolution statistics plots from JSON data files or folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s saved_stats/very_first_implementation.json
  %(prog)s saved_stats/early_results_6_confs
        """,
    )
    parser.add_argument(
        "path", help="Path to a JSON file or folder containing JSON files"
    )

    args = parser.parse_args()

    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path '{path}' does not exist.")
        sys.exit(1)

    if path.is_file():
        # Process single file
        if path.suffix.lower() != ".json":
            print(f"Error: File '{path}' is not a JSON file.")
            sys.exit(1)

        config_name = path.stem
        process_single_file(path, config_name=config_name)

    elif path.is_dir():
        # Process folder
        process_folder(path)

    else:
        print(f"Error: '{path}' is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
