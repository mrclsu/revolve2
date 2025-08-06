#!/usr/bin/env python3
"""
Statistical Analysis Script for Runtime Data

This script analyzes runtime data from a JSONL file and performs:
1. Normality tests for spider and random robots at 100 robot count
2. Appropriate statistical test (Welch t-test or Mann-Whitney U) comparing spider vs random robot performance
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List


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


def check_normality(data: List[float], name: str) -> Dict:
    """
    Perform normality test using Shapiro-Wilk test.

    Args:
        data: List of runtime values
        name: Name of the dataset for reporting

    Returns:
        Dictionary with test results
    """
    print(f"\n=== Normality Test for {name} ===")

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    print(f"Shapiro-Wilk Test:")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  P-value: {shapiro_p:.4f}")
    print(f"  Normal distribution: {'Yes' if shapiro_p > 0.05 else 'No'}")

    return {
        "shapiro_p": shapiro_p,
        "is_normal": shapiro_p > 0.05,
    }


def run_test(x: List[float], y: List[float], metric_name: str) -> Dict:
    """
    Run appropriate statistical test based on normality of data.

    Args:
        x: First group data (spider robots)
        y: Second group data (random robots)
        metric_name: Name of the metric being tested

    Returns:
        Dictionary with test results
    """
    print(f"\n=== Statistical Test: {metric_name} ===")

    # Check normality for both samples
    normal_x = stats.shapiro(x)[1] > 0.05
    normal_y = stats.shapiro(y)[1] > 0.05

    print(f"Spider robots normally distributed: {normal_x}")
    print(f"Random robots normally distributed: {normal_y}")

    # Decide test
    if normal_x and normal_y:
        # Both groups are normal - use Welch t-test
        test_result = stats.ttest_ind(x, y, equal_var=False)
        test_used = "Welch t-test"
        print(f"Using: {test_used}")
    else:
        # At least one group is not normal - use Mann-Whitney U
        test_result = stats.mannwhitneyu(x, y, alternative="two-sided")
        test_used = "Mann-Whitney U"
        print(f"Using: {test_used}")

    # Get test statistics
    if test_used == "Welch t-test":
        statistic = test_result.statistic
        p_value = test_result.pvalue
    else:
        statistic = test_result.statistic
        p_value = test_result.pvalue

    print(f"Test statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    # Calculate effect size
    if test_used == "Welch t-test":
        # Cohen's d for t-test
        pooled_std = np.sqrt(
            ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
            / (len(x) + len(y) - 2)
        )
        effect_size = (np.mean(x) - np.mean(y)) / pooled_std
        effect_size_name = "Cohen's d"
    else:
        # Rank-biserial correlation for Mann-Whitney U
        effect_size = 1 - (2 * test_result.statistic) / (len(x) * len(y))
        effect_size_name = "Rank-biserial correlation"

    print(f"Effect size ({effect_size_name}): {effect_size:.4f}")

    # Interpret effect size
    if abs(effect_size) < 0.1:
        effect_interpretation = "Negligible"
    elif abs(effect_size) < 0.3:
        effect_interpretation = "Small"
    elif abs(effect_size) < 0.5:
        effect_interpretation = "Medium"
    else:
        effect_interpretation = "Large"

    print(f"Effect size interpretation: {effect_interpretation}")

    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    print(f"  Spider robots (n={len(x)}):")
    print(f"    Mean: {np.mean(x):.4f}s")
    print(f"    Std: {np.std(x):.4f}s")
    print(f"    Median: {np.median(x):.4f}s")
    print(f"    Min: {np.min(x):.4f}s")
    print(f"    Max: {np.max(x):.4f}s")

    print(f"  Random robots (n={len(y)}):")
    print(f"    Mean: {np.mean(y):.4f}s")
    print(f"    Std: {np.std(y):.4f}s")
    print(f"    Median: {np.median(y):.4f}s")
    print(f"    Min: {np.min(y):.4f}s")
    print(f"    Max: {np.max(y):.4f}s")

    return {
        "metric": metric_name,
        "test": test_used,
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "effect_size": effect_size,
        "effect_size_name": effect_size_name,
        "effect_interpretation": effect_interpretation,
        "spider_mean": np.mean(x),
        "random_mean": np.mean(y),
        "normal_x": normal_x,
        "normal_y": normal_y,
    }


def create_normality_plots(
    spider_data: List[float],
    random_data: List[float],
    output_dir: str = "statistical_plots",
):
    """
    Create normality plots (Q-Q plots and histograms) for the data.

    Args:
        spider_data: Runtime data for spider robots
        random_data: Runtime data for random robots
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Normality Analysis for 100 Robots", fontsize=16)

    # Spider robots
    # Q-Q plot
    stats.probplot(spider_data, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title("Spider Robots - Q-Q Plot")
    axes[0, 0].grid(True, alpha=0.3)

    # Histogram
    axes[0, 1].hist(spider_data, bins=10, alpha=0.7, edgecolor="black")
    axes[0, 1].set_title("Spider Robots - Histogram")
    axes[0, 1].set_xlabel("Runtime (seconds)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Random robots
    # Q-Q plot
    stats.probplot(random_data, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Random Robots - Q-Q Plot")
    axes[1, 0].grid(True, alpha=0.3)

    # Histogram
    axes[1, 1].hist(random_data, bins=10, alpha=0.7, edgecolor="black")
    axes[1, 1].set_title("Random Robots - Histogram")
    axes[1, 1].set_xlabel("Runtime (seconds)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/normality_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated normality plots: {output_dir}/normality_analysis.png")


def main():
    """Main function to run the statistical analysis."""
    parser = argparse.ArgumentParser(
        description="Perform statistical analysis on runtime data"
    )
    parser.add_argument("input_file", help="Path to the JSONL input file")
    parser.add_argument(
        "--output-dir",
        default="statistical_plots",
        help="Output directory for plots (default: statistical_plots)",
    )
    parser.add_argument(
        "--robot-count",
        type=int,
        default=100,
        help="Robot count to analyze (default: 100)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        return

    try:
        # Load data
        print(f"Loading data from {args.input_file}...")
        data = load_data(args.input_file)

        if not data:
            print("Error: No data found in the input file.")
            return

        # Check if we have data for the specified robot count
        if args.robot_count not in data.get(
            "spider", {}
        ) or args.robot_count not in data.get("random", {}):
            print(f"Error: No data found for {args.robot_count} robots.")
            return

        spider_data = data["spider"][args.robot_count]
        random_data = data["random"][args.robot_count]

        print(f"Analyzing data for {args.robot_count} robots:")
        print(f"  Spider robots: {len(spider_data)} samples")
        print(f"  Random robots: {len(random_data)} samples")

        # Perform normality tests
        spider_normality = check_normality(spider_data, "Spider Robots")
        random_normality = check_normality(random_data, "Random Robots")

        # Run appropriate statistical test
        test_results = run_test(spider_data, random_data, "Runtime Performance")

        # Create normality plots
        create_normality_plots(spider_data, random_data, args.output_dir)

        # Summary
        print(f"\n=== SUMMARY ===")
        print(f"Spider robots normally distributed: {spider_normality['is_normal']}")
        print(f"Random robots normally distributed: {random_normality['is_normal']}")
        print(f"Test used: {test_results['test']}")
        print(f"Significant difference between groups: {test_results['significant']}")
        print(
            f"Effect size: {test_results['effect_interpretation']} ({test_results['effect_size_name']} = {test_results['effect_size']:.4f})"
        )

        if test_results["significant"]:
            faster_robot = (
                "Spider"
                if test_results["spider_mean"] < test_results["random_mean"]
                else "Random"
            )
            print(f"Faster robot type: {faster_robot}")

        print(f"\nAnalysis complete! Plots saved in '{args.output_dir}' directory.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        return


if __name__ == "__main__":
    main()
