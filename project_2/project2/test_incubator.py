#!/usr/bin/env python3
"""Test script for the Incubator class."""

import logging
import multineat
import numpy as np

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed

from incubator import Incubator
from genotype import Genotype


def main() -> None:
    """Test the incubator functionality."""
    # Set up logging
    setup_logging()

    # Set up random number generator
    rng = make_rng_time_seed()

    # Create innovation databases
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # Test parameters
    population_size = 10
    training_budget = 10  # Small budget for testing

    logging.info("Testing Incubator with RevDE pretraining...")
    logging.info(f"Population size: {population_size}")
    logging.info(f"Training budget: {training_budget}")

    # Create incubator
    incubator = Incubator(
        population_size=population_size,
        training_budget=training_budget,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        headless=True,
        num_simulators=1,  # Use single simulator for testing
    )

    # Run incubation process
    pretrained_population = incubator.incubate()

    # Display results
    logging.info("Incubation completed!")
    logging.info(f"Final population size: {len(pretrained_population)}")

    # Verify population size is maintained
    assert len(pretrained_population) == population_size, (
        f"Population size changed from {population_size} to {len(pretrained_population)}"
    )

    # Verify that all individuals have the same body structure (brain-only training)
    initial_genotypes = [
        Genotype(body=ind.body, brain=ind.brain) for ind in incubator.population
    ]
    initial_bodies = [g.body.genotype.Serialize() for g in initial_genotypes]
    final_bodies = [
        ind.genotype.body.genotype.Serialize() for ind in pretrained_population
    ]

    bodies_unchanged = all(
        initial == final for initial, final in zip(initial_bodies, final_bodies)
    )
    logging.info(f"Bodies unchanged during incubation: {bodies_unchanged}")
    assert bodies_unchanged, "Bodies should remain unchanged during brain-only training"

    for i, individual in enumerate(pretrained_population):
        logging.info(f"Individual {i}: fitness = {individual.fitness:.4f}")

    # Calculate fitness statistics
    fitnesses = [ind.fitness for ind in pretrained_population]
    mean_fitness = np.mean(fitnesses)
    std_fitness = np.std(fitnesses)
    max_fitness = np.max(fitnesses)
    min_fitness = np.min(fitnesses)

    logging.info(f"Fitness statistics:")
    logging.info(f"  Mean: {mean_fitness:.4f}")
    logging.info(f"  Std:  {std_fitness:.4f}")
    logging.info(f"  Max:  {max_fitness:.4f}")
    logging.info(f"  Min:  {min_fitness:.4f}")

    logging.info("✅ All tests passed! Incubator is working correctly.")


if __name__ == "__main__":
    main()
