"""Configuration parameters for this example."""

DATABASE_FILE = "database.sqlite"
NUM_SIMULATORS = 8
POPULATION_SIZE = 4  # Setting this to 1 will result in warnings and faulty diversity measures, as you need more than 1 individual for that.
EVALUATE = True
VISUALIZE_MAP = False  # Be careful when setting this to true when POPULATION_size > 1, as you will get plots for each individual.

# Incubator settings
USE_INCUBATOR = True  # Whether to use the incubator for pretraining
INCUBATOR_TRAINING_BUDGET = 10  # Number of RevDE iterations for pretraining
