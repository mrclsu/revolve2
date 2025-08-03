"""Main script for the example."""

import logging

from project2.config import Config
import multineat
from project2.genotype import Genotype
from project2.individual import Individual


from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from project2.stats import Statistics
from project2.incubator import Incubator


from .robot_evolution import ModularRobotEvolution
from .parent_selector import ParentSelector
from .survivor_selector import SurvivorSelector
from .evaluator import Evaluator
from .crossover_reproducer import CrossoverReproducer


def run_standard_setup(
    config: Config, stats_folder: str = "stats/standard_setup"
) -> None:
    """Run the program."""
    # Set up logging.
    setup_logging(file_name="log.txt")

    # Set up the random number generator.
    rng = make_rng_time_seed()

    # CPPN innovation databases.
    # If you don't understand CPPN, just know that a single database is shared in the whole evolutionary process.
    # One for body, and one for brain.
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    """
    Here we initialize the components used for the evolutionary process.

    - evaluator: Allows us to evaluate a population of modular robots.
    - parent_selector: Allows us to select parents from a population of modular robots.
    - survivor_selector: Allows us to select survivors from a population.
    - crossover_reproducer: Allows us to generate offspring from parents.
    - modular_robot_evolution: The evolutionary process as a object that can be iterated.
    """
    evaluator = Evaluator(
        headless=True,
        num_simulators=config.NUM_SIMULATORS,
        plane_size=config.LIMITS.calculate_plane_size(),
        movement_weight=config.MOVEMENT_WEIGHT,
        fitness_function_algorithm=config.FITNESS_FUNCTION_ALGORITHM,
    )

    # TODO: figure out offspring size, currently it's half the population size
    parent_selector = ParentSelector(
        offspring_size=config.POPULATION_SIZE // 2, rng=rng
    )
    survivor_selector = SurvivorSelector(rng=rng)
    crossover_reproducer = CrossoverReproducer(
        rng=rng, innov_db_body=innov_db_body, innov_db_brain=innov_db_brain
    )

    modular_robot_evolution = ModularRobotEvolution(
        parent_selection=parent_selector,
        survivor_selection=survivor_selector,
        evaluator=evaluator,
        reproducer=crossover_reproducer,
    )

    # Create an initial population as we cant start from nothing.
    logging.info("Generating initial population.")
    incubated_population = Incubator(
        population_size=config.POPULATION_SIZE,
        training_budget=config.INCUBATOR_TRAINING_BUDGET,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        num_simulators=config.NUM_SIMULATORS,
    ).incubate()

    initial_genotypes = [individual.genotype for individual in incubated_population]

    logging.info("Evaluating initial population.")
    initial_fitnesses, initial_all_fitness_metrics = evaluator.evaluate(
        initial_genotypes
    )

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness, 0, fitness_metrics=all_fitness_metrics)
        for genotype, fitness, all_fitness_metrics in zip(
            initial_genotypes,
            initial_fitnesses,
            initial_all_fitness_metrics,
            strict=True,
        )
    ]

    for individual in population:
        individual.develop()

    # Set the current generation to 0.
    generation_index = 0

    stats = Statistics(stats_folder)

    # Start the actual optimization process.
    logging.info("Start optimization process.")
    while generation_index < config.ITERATIONS:
        logging.info(f"Generation {generation_index} / {config.ITERATIONS}.")

        for individual in population:
            if individual.get_robot_uuid() is None:
                logging.warning("none found in population")

        population = modular_robot_evolution.step(
            population,
            generation_index=generation_index,
            stats=stats,
        )

        stats.flush_to_json(f"generation_{generation_index}")

        generation_index += 1
