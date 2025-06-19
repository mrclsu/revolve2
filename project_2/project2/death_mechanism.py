"""Death mechanism implementations for population management."""

import logging
from enum import Enum


from project2.individual import Individual


class DeathMechanism(Enum):
    """Enum for different death mechanism strategies."""

    MAX_AGE = "max_age"
    LOWEST_FITNESS = "lowest_fitness"


def apply_max_age_death(
    population: list[Individual],
    current_generation: int,
    max_population_size: int,
    min_population_size: int,
    max_age: int,
) -> list[Individual]:
    dead_individuals: list[Individual] = []
    dead_robot_uuids = set()

    if len(population) > min_population_size:
        for ind in population:
            if (ind.initial_generation + current_generation) > max_age:
                dead_individuals.append(ind)
                dead_robot_uuids.add(ind.get_robot_uuid())
            if len(population) - len(dead_individuals) < min_population_size:
                break

    remaining_population = [
        ind for ind in population if ind.get_robot_uuid() not in dead_robot_uuids
    ]
    if len(remaining_population) > max_population_size:
        remaining_population.sort(key=lambda x: x.initial_generation)
        excess_count = len(remaining_population) - max_population_size
        additional_removals = remaining_population[:excess_count]
        dead_individuals.extend(additional_removals)

    logging.info(f"Age-based death: removing {len(dead_individuals)} individuals")
    return dead_individuals


def apply_lowest_fitness_death(
    population: list[Individual],
    current_generation: int,
    max_population_size: int,
    min_population_size: int,
) -> list[Individual]:
    eligible_for_death = [
        ind
        for ind in population
        if ind.initial_generation != current_generation and ind.fitness is not None
    ]

    target_removals = len(population) - max_population_size
    if target_removals <= 0:
        logging.info("Fitness-based death: no removals needed")
        return []

    max_removals = len(population) - min_population_size
    actual_removals = min(target_removals, max_removals, len(eligible_for_death))

    if actual_removals <= 0:
        logging.info("Fitness-based death: no eligible individuals for removal")
        return []

    eligible_for_death.sort(key=lambda x: (x.fitness, -x.initial_generation))

    individuals_to_remove = eligible_for_death[:actual_removals]

    fitness_scores = [ind.fitness for ind in individuals_to_remove]
    logging.info(
        f"Fitness-based death: removing {len(individuals_to_remove)} individuals "
        f"with fitness scores: {fitness_scores}"
    )

    return individuals_to_remove


def apply_death_mechanism(
    population: list[Individual],
    current_generation: int,
    death_mechanism: DeathMechanism,
    max_population_size: int,
    min_population_size: int,
    max_age: int,
) -> list[Individual]:
    logging.info(f"Applying death mechanism: {death_mechanism.value}")
    if death_mechanism == DeathMechanism.MAX_AGE:
        return apply_max_age_death(
            population=population,
            current_generation=current_generation,
            max_population_size=max_population_size,
            min_population_size=min_population_size,
            max_age=max_age,
        )
    elif death_mechanism == DeathMechanism.LOWEST_FITNESS:
        return apply_lowest_fitness_death(
            population=population,
            current_generation=current_generation,
            max_population_size=max_population_size,
            min_population_size=min_population_size,
        )
    else:
        raise ValueError(f"Unknown death mechanism: {death_mechanism}")
