from enum import Enum
import logging
import random
from revolve2.standards.genotype import Genotype
from project2.individual import Individual

import numpy as np
import multineat

from revolve2.experimentation.evolution.abstract_elements import Reproducer
from revolve2.experimentation.rng import make_rng_time_seed


THRESHOLD = 0.1


class MateSelectionStrategy(Enum):
    OPPOSITES = 1
    SIMILAR = 2
    MAX_FITNESS = 3


def mate_decision(
    strategy: MateSelectionStrategy, individual1: Individual, individual2: Individual
):
    logging.info(
        f"Strategy: {strategy}, individual1: {individual1.fitness}, individual2: {individual2.fitness}"
    )
    if strategy == MateSelectionStrategy.OPPOSITES:
        return abs(individual1.fitness - individual2.fitness) > THRESHOLD
    elif strategy == MateSelectionStrategy.SIMILAR:
        return abs(individual1.fitness - individual2.fitness) < THRESHOLD
    elif strategy == MateSelectionStrategy.MAX_FITNESS:
        return (individual1.fitness - individual2.fitness) < 0


def reproduce(parent1: Individual, parent2: Individual, rng: np.random.Generator):
    offspring = [Genotype.crossover(parent1.genotype, parent2.genotype, rng)]
    return Individual(genotype=offspring[0], fitness=0.0)
