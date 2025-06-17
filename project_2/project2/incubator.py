import logging
from revolve2.simulation.scene.vector2.vector2 import Vector2
from project2.individual import Individual
from project2.genotype import Genotype

import multineat
import numpy as np
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from project2.utils.helpers import initialize_local_simulator
from project2.simulation_result import SimulationResult


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

        # RevDE algorithm parameters
        self.F = 0.5  # Scaling factor for differential mutation
        self.CR = 0.9  # Crossover probability
        self.transformation_prob = (
            0.3  # Probability of applying reversible transformation
        )

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
    ) -> Genotype:
        """
        Perform differential mutation using RevDE strategy, but only on the brain.
        The body remains unchanged from the target individual.

        :param target_idx: Index of target individual
        :param population: Current population
        :return: Genotype with mutated brain but original body
        """
        # Select three random individuals different from target
        candidates = [i for i in range(len(population)) if i != target_idx]
        if len(candidates) < 3:
            # If population too small, use regular brain mutation
            mutated = population[target_idx].mutate_brain_only(
                innov_db_brain=self.innov_db_brain,
                rng=self.rng,
            )
            return mutated

        a, b, c = self.rng.choice(candidates, 3, replace=False)

        # Keep the original body from target, only modify brain
        target_body = population[target_idx].body

        # Create donor brain by combining brains from a, b, c
        # Start with brain from 'a'
        donor_brain = population[a].brain

        # Apply differential scaling by using brain crossover with modified probability
        if self.rng.random() < self.F:
            # Crossover between brains b and c, then with a
            bc_cross = Genotype.crossover_brains_only(
                population[b], population[c], self.rng
            )
            donor_cross = Genotype.crossover_brains_only(
                Genotype(body=target_body, brain=donor_brain),
                Genotype(body=target_body, brain=bc_cross.brain),
                self.rng,
            )
            donor_brain = donor_cross.brain

        # Apply additional brain mutation with some probability
        if self.rng.random() < self.transformation_prob:
            temp_genotype = Genotype(body=target_body, brain=donor_brain)
            mutated_genotype = temp_genotype.mutate_brain_only(
                innov_db_brain=self.innov_db_brain,
                rng=self.rng,
            )
            donor_brain = mutated_genotype.brain

        return Genotype(body=target_body, brain=donor_brain)

    def _crossover_step(self, target: Genotype, donor: Genotype) -> Genotype:
        """
        Perform crossover between target and donor, but only on brains.
        The body from target is always kept.

        :param target: Target genotype (body will be kept)
        :param donor: Donor genotype
        :return: Trial genotype with target's body
        """
        if self.rng.random() < self.CR:
            return Genotype.crossover_brains_only(target, donor, self.rng)
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
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        simulation_result = SimulationResult(scene_states)
        return simulation_result.fitness(robots)

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
                donor = self._differential_mutation_brain_only(i, genotypes)

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
