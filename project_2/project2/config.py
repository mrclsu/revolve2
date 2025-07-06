"""Configuration parameters for this example."""

import json
from pathlib import Path
from typing import Dict, Any

from project2.utils.field_limits import FieldLimits
from project2.simulation_result import FitnessFunctionAlgorithm
from project2.mate_selection import MateSelectionStrategy
from project2.death_mechanism import DeathMechanism


class Config:
    """Configuration class that loads settings from a JSON file."""

    def __init__(self, config_name: str = "config1"):
        """Initialize configuration from JSON file.

        Args:
            config_name: Name of the config file (without .json extension) or full path
        """
        self._config_data = self._load_config(config_name)
        self._initialize_properties()

    def _load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from JSON file.

        Args:
            config_name: Name of the config file (without .json extension) or full path

        Returns:
            Dictionary containing all configuration parameters
        """
        # Handle both config name and full path
        if config_name.endswith(".json"):
            config_file = Path(config_name)
        else:
            # Get the directory of this file
            config_dir = Path(__file__).parent / "configs"
            config_file = config_dir / f"{config_name}.json"

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            config_data = json.load(f)

        return config_data

    def _initialize_properties(self) -> None:
        """Initialize all configuration properties from the loaded data."""
        # Basic settings
        self.DATABASE_FILE = self._config_data["database_file"]
        self.NUM_SIMULATORS = self._config_data["num_simulators"]
        self.EVALUATE = self._config_data["evaluate"]
        self.SIMULATION_HEADLESS = self._config_data["simulation_headless"]
        self.VISUALIZE_MAP = self._config_data["visualize_map"]

        # Incubator settings
        self.INCUBATOR_TRAINING_BUDGET = self._config_data["incubator_training_budget"]

        # Population settings
        self.POPULATION_SIZE = self._config_data["population_size"]
        self.MIN_POPULATION_SIZE = self._config_data["min_population_size"]
        self.MAX_POPULATION_SIZE = self._config_data["max_population_size"]
        self.MAX_AGE = self._config_data["max_age"]

        # Simulation settings
        self.ITERATIONS = self._config_data["iterations"]
        self.FIELD_X_MIN = self._config_data["field_x_min"]
        self.FIELD_X_MAX = self._config_data["field_x_max"]
        self.FIELD_Y_MIN = self._config_data["field_y_min"]
        self.FIELD_Y_MAX = self._config_data["field_y_max"]

        self.LIMITS = FieldLimits(
            self.FIELD_X_MIN, self.FIELD_X_MAX, self.FIELD_Y_MIN, self.FIELD_Y_MAX
        )

        # Fitness and mating settings (convert string enums to actual enums)
        self.FITNESS_FUNCTION_ALGORITHM = FitnessFunctionAlgorithm[
            self._config_data["fitness_function_algorithm"]
        ]
        self.MATE_SELECTION_STRATEGY = MateSelectionStrategy[
            self._config_data["mate_selection_strategy"]
        ]
        self.MATE_SELECTION_THRESHOLD = self._config_data["mate_selection_threshold"]

        self.MATING_THRESHOLD = self._config_data["mating_threshold"]
        self.MATING_COOLDOWN = self._config_data["mating_cooldown"]

        # Death mechanism settings (convert string enum to actual enum)
        self.DEATH_MECHANISM = DeathMechanism[self._config_data["death_mechanism"]]

        # Export config for debugging/logging
        self.CURRENT_CONFIG = self._config_data

    def get_config_data(self) -> Dict[str, Any]:
        """Get the raw configuration data."""
        return self._config_data.copy()
