"""Configuration parameters for this example."""

POPULATION_SIZE = 50
OFFSPRING_SIZE = 25
NUM_GENERATIONS = 200
NUM_PARAMETERS = 9
MUTATE_STD = 0.05
DATABASE_FILE = "database.sqlite"
NUM_SIMULATORS = 8
POPULATION_SIZE = 4  # Setting this to 1 will result in warnings and faulty diversity measures, as you need more than 1 individual for that.
EVALUATE = True
VISUALIZE_MAP = False  # Be careful when setting this to true when POPULATION_size > 1, as you will get plots for each individual.
