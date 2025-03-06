import random
from revolve2.standards.genotype import Genotype

import numpy as np

from revolve2.experimentation.evolution.abstract_elements import Reproducer
from revolve2.experimentation.rng import make_rng_time_seed



def mate_decision():
    if random.random() > 0.01:
        decision = True 
    else:
        decision = False

    return decision

def reproduce(parent1, parent2, rng):
    #rng = make_rng_time_seed()
    offspring = [Genotype.crossover(parent1.genotype, parent2.genotype, rng).mutate(rng)]
    return offspring

