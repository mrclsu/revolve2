import random
from revolve2.standards.genotype import Genotype
from individual import Individual

import numpy as np
import multineat

from revolve2.experimentation.evolution.abstract_elements import Reproducer
from revolve2.experimentation.rng import make_rng_time_seed


def mate_decision():
    if random.random() > 0.01:
        decision = True
    else:
        decision = False

    return decision


def reproduce(parent1, parent2, rng):
    rng_multi = multineat.RNG()
    offspring = [Genotype.crossover(parent1.genotype, parent2.genotype, rng_multi)]
    # offspring = [Genotype.crossover(parent1.genotype, parent2.genotype, rng_multi).mutate(rng)]
    return Individual(genotype=offspring[0], fitness=0.0)
