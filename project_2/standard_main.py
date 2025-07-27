from project2.config import Config
from project2.standard_setup.standard_setup import run_standard_setup


if __name__ == "__main__":
    import argparse
    import os
    import glob

    # Dynamically get available config choices from configs/ folder
    config_dir = os.path.join(os.path.dirname(__file__), "project2", "configs")
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    available_configs = [os.path.splitext(os.path.basename(f))[0] for f in config_files]
    available_configs.sort()  # Sort for consistent ordering

    parser = argparse.ArgumentParser(
        description="Run simulation with different configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=available_configs,
        default="config1"
        if "config1" in available_configs
        else (available_configs[0] if available_configs else "config1"),
        help=f"Config name to use. Available configs: {', '.join(available_configs)}",
    )
    args = parser.parse_args()

    # Create a Config instance with the specified config name
    config = Config(args.config)

    run_standard_setup(config, f"stats/standard_setup/{args.config}")
