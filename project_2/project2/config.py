"""Configuration parameters for this example."""

from project2.utils.field_limits import FieldLimits
from project2.simulation_result import FitnessFunctionAlgorithm
from project2.mate_selection import MateSelectionStrategy

DATABASE_FILE = "database.sqlite"
NUM_SIMULATORS = 8
EVALUATE = True
SIMULATION_HEADLESS = True
VISUALIZE_MAP = False  # Be careful when setting this to true when POPULATION_size > 1, as you will get plots for each individual.

# Incubator settings
INCUBATOR_TRAINING_BUDGET = (
    0  # Number of RevDE iterations for pretraining, 0 disables pretraining
)

# Population settings
POPULATION_SIZE = 30  # This is the size of the initial population
MIN_POPULATION_SIZE = 25  # This is the minimum size of the population
MAX_POPULATION_SIZE = 50  # This is the maximum size of the population
MAX_AGE = 20  # This is the maximum age of an individual

# Mating settings
MATING_THRESHOLD = 1.2  # This is the threshold for mating

# Simulation settings
ITERATIONS = 10
FIELD_X_MIN, FIELD_X_MAX = -10, 10  # Adjust based on your simulation size
FIELD_Y_MIN, FIELD_Y_MAX = -10, 10

LIMITS = FieldLimits(FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)

# Fitness and mating settings
FITNESS_FUNCTION_ALGORITHM = FitnessFunctionAlgorithm.XY_DISPLACEMENT
MATE_SELECTION_STRATEGY = MateSelectionStrategy.MAX_FITNESS
