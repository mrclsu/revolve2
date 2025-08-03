import logging
from revolve2.simulation.scene.vector2.vector2 import Vector2
from project2.individual import Individual
from project2.genotype import Genotype

import multineat
import numpy as np
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from project2.utils.helpers import initialize_local_simulator
from project2.simulation_result import SimulationResult, FitnessFunctionAlgorithm


class Incubator:
    def __init__(
        self,
        population_size: int,
        training_budget: int,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
        headless: bool = True,
        num_simulators: int = 1,
        plane_size: float = 10.0,
    ):
        self.population_size = population_size
        self.training_budget = training_budget
        self.innov_db_body = innov_db_body
        self.innov_db_brain = innov_db_brain
        self.rng = rng

        self._simulator = initialize_local_simulator(
            plane_size, headless=headless, num_simulators=num_simulators
        )

        self.plane_size = plane_size

        # RevDE algorithm parameters (matching original implementation)
        self.F = 0.5  # Scaling factor for differential mutation
        self.CR = 0.9  # Crossover probability
        # Note: Original RevDE always applies reversible transformation (no probability)

        self.initialize_population()

    def initialize_population(self) -> list[Genotype]:
        self.population = [
            Genotype.random(
                innov_db_body=self.innov_db_body,
                innov_db_brain=self.innov_db_brain,
                rng=self.rng,
            )
            for _ in range(self.population_size)
        ]
        return self.population

    def _differential_mutation_brain_only(
        self, target_idx: int, population: list[Genotype]
    ) -> tuple[Genotype, list[Genotype]]:
        """
        Perform differential mutation using RevDE strategy, but only on the brain.
        The body remains unchanged from the target individual.

        Implements the reversible linear transformation from the original RevDE paper:
        y_1 = x_1 + F * (x_2 - x_3)
        y_2 = x_2 + F * (x_3 - y_1)
        y_3 = x_3 + F * (y_1 - y_2)

        :param target_idx: Index of target individual
        :param population: Current population
        :return: Tuple of (primary donor genotype, list of all donors)
        """
        # Select three random individuals different from target
        candidates = [i for i in range(len(population)) if i != target_idx]
        if len(candidates) < 3:
            # If population too small, use regular brain mutation
            mutated = population[target_idx].mutate_brain_only(
                innov_db_brain=self.innov_db_brain,
                rng=self.rng,
            )
            return mutated, [mutated]

        # Select three individuals: a, b, c
        a, b, c = self.rng.choice(candidates, 3, replace=False)

        # Get the corresponding genotypes
        x1 = population[target_idx]  # target
        x2 = population[a]  # first random
        x3 = population[b]  # second random

        # Keep target body unchanged
        target_body = x1.body

        # Apply RevDE reversible transformation (always, as per original paper)
        # Apply RevDE reversible transformation - BRAIN ONLY
        # For brain genotypes, we approximate the differential operation using
        # weighted crossover to simulate the linear combination

        # Step 1: y_1 = x_1 + F * (x_2 - x_3) - brain only
        x2_x3_cross_brain = Genotype.crossover_brains_only(x2, x3, self.rng)
        if self.rng.random() < self.F:
            y1_brain = Genotype.crossover_brains_only(
                x1, x2_x3_cross_brain, self.rng
            ).brain
        else:
            y1_brain = x1.brain
        y1 = Genotype(body=target_body, brain=y1_brain)

        # Step 2: y_2 = x_2 + F * (x_3 - y_1) - brain only
        x3_y1_cross_brain = Genotype.crossover_brains_only(x3, y1, self.rng)
        if self.rng.random() < self.F:
            y2_brain = Genotype.crossover_brains_only(
                x2, x3_y1_cross_brain, self.rng
            ).brain
        else:
            y2_brain = x2.brain
        y2 = Genotype(body=target_body, brain=y2_brain)

        # Step 3: y_3 = x_3 + F * (y_1 - y_2) - brain only
        y1_y2_cross_brain = Genotype.crossover_brains_only(y1, y2, self.rng)
        if self.rng.random() < self.F:
            y3_brain = Genotype.crossover_brains_only(
                x3, y1_y2_cross_brain, self.rng
            ).brain
        else:
            y3_brain = x3.brain
        y3 = Genotype(body=target_body, brain=y3_brain)

        return y1, [y1, y2, y3]

    def _crossover_step(self, target: Genotype, donor: Genotype) -> Genotype:
        """
        Perform crossover between target and donor, but only on brains.
        The body from target is always kept.

        :param target: Target genotype (body will be kept)
        :param donor: Donor genotype
        :return: Trial genotype with target's body
        """
        if self.rng.random() < self.CR:
            # Brain-only crossover - explicitly preserve target body
            crossed_brain = Genotype.crossover_brains_only(
                target, donor, self.rng
            ).brain
            return Genotype(body=target.body, brain=crossed_brain)
        else:
            return target

    def _evaluate_fitness_batch(self, genotypes: list[Genotype]) -> list[float]:
        """
        Evaluate fitness of multiple genotypes efficiently using batch simulation.

        :param genotypes: List of genotypes to evaluate
        :return: List of fitness values (xy displacements)
        """
        # Develop all genotypes into robots
        robots = [genotype.develop(visualize=False) for genotype in genotypes]

        # Create scenes for all robots
        scenes = []

        for robot in robots:
            scene = ModularRobotScene(
                terrain=terrains.flat(Vector2([self.plane_size, self.plane_size]))
            )
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes
        all_scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        fitness_values = []
        for robot, scene_states in zip(robots, all_scene_states):
            sim_res = SimulationResult(scene_states, plane_size=self.plane_size)
            fitness_values.append(
                sim_res.fitness([robot], FitnessFunctionAlgorithm.MAX_DISTANCE)
            )

        return fitness_values

    def incubate(self) -> list[Individual]:
        """
        Pretrain the population using RevDE algorithm, focusing only on brain training.
        Bodies remain fixed during the incubation process.

        :return: List of pretrained individuals with optimized brains
        """
        logging.info(
            f"Starting RevDE brain-only pretraining with {self.training_budget} iterations..."
        )

        # Convert genotypes to individuals with initial fitness using batch evaluation
        logging.info("Evaluating initial population...")
        initial_fitnesses = self._evaluate_fitness_batch(self.population)
        individuals = [
            Individual(genotype, fitness)
            for genotype, fitness in zip(self.population, initial_fitnesses)
        ]

        # RevDE main loop
        for iteration in range(self.training_budget):
            if iteration % 5 == 0:
                logging.info(f"RevDE iteration {iteration}/{self.training_budget}")

            new_individuals: list[Individual] = []
            trial_genotypes: list[Genotype] = []

            # Generate all trial genotypes first
            for i in range(len(individuals)):
                # Extract genotypes for differential mutation
                genotypes = [ind.genotype for ind in individuals]

                # Differential mutation (brain only)
                donor, all_donors = self._differential_mutation_brain_only(i, genotypes)

                # Crossover
                trial = self._crossover_step(individuals[i].genotype, donor)
                trial_genotypes.append(trial)

            # Batch evaluate all trial genotypes
            trial_fitnesses = self._evaluate_fitness_batch(trial_genotypes)

            # Selection - compare trials with targets
            for i, (trial, trial_fitness) in enumerate(
                zip(trial_genotypes, trial_fitnesses)
            ):
                if trial_fitness >= individuals[i].fitness:
                    # Trial is better or equal, replace target
                    new_individuals.append(Individual(trial, trial_fitness))
                else:
                    # Keep original
                    new_individuals.append(individuals[i])

            individuals = new_individuals

        logging.info("RevDE pretraining completed!")
        return individuals
