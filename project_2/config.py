"""Configuration parameters for this example."""

DATABASE_FILE = "database.sqlite"
NUM_SIMULATORS = 8
EVALUATE = True
VISUALIZE_MAP = False  # Be careful when setting this to true when POPULATION_size > 1, as you will get plots for each individual.

# Incubator settings
INCUBATOR_TRAINING_BUDGET = (
    100  # Number of RevDE iterations for pretraining, 0 disables pretraining
)

# Population settings
POPULATION_SIZE = 30  # This is the size of the initial population
MIN_POPULATION_SIZE = 25  # This is the minimum size of the population
MAX_POPULATION_SIZE = 50  # This is the maximum size of the population
MAX_AGE = 20  # This is the maximum age of an individual

# Mating settings
MATING_THRESHOLD = 1.2  # This is the threshold for mating
