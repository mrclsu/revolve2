from typing import Any
import logging

import numpy as np
from project2.individual import Individual

from revolve2.experimentation.evolution.abstract_elements import Selector
from revolve2.experimentation.optimization.ea import population_management, selection

from project2.stats import Statistics


class SurvivorSelector(Selector):
    """Selector class for survivor selection."""

    rng: np.random.Generator

    def __init__(self, rng: np.random.Generator) -> None:
        """
        Initialize the parent selector.

        :param rng: The rng generator.
        """
        self.rng = rng

    def select(
        self, population: list[Individual], **kwargs: Any
    ) -> tuple[list[Individual], dict[str, Any]]:
        """
        Select survivors using a tournament.

        :param population: The population the parents come from.
        :param kwargs: The offspring, with key 'offspring_population'.
        :returns: A newly created population.
        :raises ValueError: If the population is empty.
        """
        offspring = kwargs.get("children")
        offspring_fitness = kwargs.get("child_task_performance")
        offspring_all_fitness_metrics = kwargs.get("child_all_fitness_metrics")
        generation_index = kwargs.get("generation_index")
        stats: Statistics = kwargs.get("stats")
        if (
            offspring is None
            or offspring_fitness is None
            or generation_index is None
            or stats is None
            or offspring_all_fitness_metrics is None
        ):
            raise ValueError(
                "No offspring was passed with positional argument 'children' and / or 'child_task_performance' and / or 'initial_generation' and / or 'stats' and / or 'child_all_fitness_metrics'."
            )

        original_survivors, offspring_survivors = population_management.steady_state(
            old_genotypes=[i.genotype for i in population],
            old_fitnesses=[i.fitness for i in population],
            new_genotypes=offspring,
            new_fitnesses=offspring_fitness,
            selection_function=lambda n,
            genotypes,
            fitnesses: selection.multiple_unique(
                selection_size=n,
                population=genotypes,
                fitnesses=fitnesses,
                selection_function=lambda _, fitnesses: selection.tournament(
                    rng=self.rng, fitnesses=fitnesses, k=2
                ),
            ),
        )

        for i in range(len(population)):
            if i not in original_survivors:
                population[i].final_generation = generation_index

        new_pop = [population[i] for i in original_survivors] + [
            Individual(
                offspring[i],
                offspring_fitness[i],
                generation_index,
                offspring_all_fitness_metrics[i],
            )
            for i in offspring_survivors
        ]

        for individual in new_pop:
            individual.develop()

        stats.add_generation(generation_index, len(new_pop))

        old_uuid_to_individual = {
            individual.get_robot_uuid(): individual for individual in population
        }
        new_uuid_to_individual = {
            individual.get_robot_uuid(): individual for individual in new_pop
        }

        if None in new_uuid_to_individual:
            logging.warning("none found in new_uuid_to_individual")

        if None in old_uuid_to_individual:
            logging.warning("none found in old_uuid_to_individual")

        uuid_to_individual = {
            **old_uuid_to_individual,
            **new_uuid_to_individual,
        }

        fitness_metrics_by_uuid = {}

        for uuid, individual in uuid_to_individual.items():
            for metric_name, metric_values in individual.fitness_metrics.items():
                if metric_name not in fitness_metrics_by_uuid:
                    fitness_metrics_by_uuid[metric_name] = {}
                fitness_metrics_by_uuid[metric_name][uuid] = metric_values[0]

        stats.track_individuals(
            uuid_to_individual, generation_index, fitness_metrics_by_uuid
        )

        return new_pop, {}
