#!/usr/bin/env python3
"""
Script to plot evolution statistics from JSON data.
Creates three plots:
1. Population size over generations
2. Population size with births/deaths overlay
3. Average age of robots across generations
"""

import json
import matplotlib.pyplot as plt
import numpy as np
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


def plot_population_size(generation_to_population_size):
    """Create a plot showing population size over generations."""
    generations = [int(gen) for gen in generation_to_population_size.keys()]
    population_sizes = [generation_to_population_size[str(gen)] for gen in generations]

    plt.figure(figsize=(12, 6))
    plt.plot(generations, population_sizes, "b-", linewidth=2, marker="o", markersize=3)
    plt.xlabel("Generation")
    plt.ylabel("Population Size")
    plt.title("Population Size Over Generations")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_population_with_births_deaths(generation_to_population_size, robot_stats):
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

    plt.title("Population Size with Births and Deaths Over Generations")
    plt.tight_layout()
    return fig


def plot_average_age(generation_to_population_size, robot_stats):
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
    plt.title("Mean, Median, and Oldest Age of Robots Across Generations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_births_deaths_only(generation_to_population_size, robot_stats):
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
    plt.title("Births and Deaths per Generation")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(
        ticks=x_pos[::5], labels=x_pos[::5]
    )  # Show ticks every 5 generations for clarity
    plt.tight_layout()
    return plt.gcf()


def print_summary_stats(generation_to_population_size, robot_stats):
    """Print summary statistics."""
    generations = [int(gen) for gen in generation_to_population_size.keys()]
    population_sizes = [generation_to_population_size[str(gen)] for gen in generations]

    births, deaths = calculate_births_deaths(robot_stats)
    avg_ages, median_ages, max_ages = calculate_age_statistics_per_generation(
        robot_stats, max(generations)
    )

    print("=== Evolution Summary Statistics ===")
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


def main():
    # Load data
    data = load_data("saved_stats/very_first_implementation.json")
    generation_to_population_size = data["generation_to_population_size"]
    robot_stats = data["robot_stats"]

    # Print summary statistics
    print_summary_stats(generation_to_population_size, robot_stats)
    print()

    # Create plots
    print("Creating population size plot...")
    fig1 = plot_population_size(generation_to_population_size)
    plt.savefig("population_size_over_time.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Creating population with births/deaths plot...")
    fig2 = plot_population_with_births_deaths(
        generation_to_population_size, robot_stats
    )
    plt.savefig("population_with_births_deaths.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Creating average age plot...")
    fig3 = plot_average_age(generation_to_population_size, robot_stats)
    plt.savefig("average_age_over_time.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Creating births and deaths plot...")
    fig4 = plot_births_deaths_only(generation_to_population_size, robot_stats)
    plt.savefig("births_deaths_per_generation.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(
        "Plots saved as 'population_size_over_time.png', 'population_with_births_deaths.png', 'average_age_over_time.png', and 'births_deaths_per_generation.png'"
    )


if __name__ == "__main__":
    main()
