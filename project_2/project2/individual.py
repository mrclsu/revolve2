"""Individual class."""

import logging
from dataclasses import dataclass, field
from uuid import UUID

from revolve2.modular_robot._modular_robot import ModularRobot

from project2.genotype import Genotype

import numpy as np
import multineat


@dataclass
class Individual:
    """An individual in a population."""

    genotype: Genotype
    fitness: float
    fitness_metrics: dict[str, float]

    initial_generation: int = 0
    final_generation: int = -1
    robot: ModularRobot | None = None

    def __init__(
        self,
        genotype: Genotype,
        fitness: float,
        initial_generation: int = 0,
        fitness_metrics: dict[str, float] = field(default_factory=dict),
    ):
        self.genotype = genotype
        self.fitness = fitness
        self.initial_generation = initial_generation
        self.fitness_metrics = fitness_metrics

    def get_robot_uuid(self) -> UUID | None:
        return self.robot.uuid if self.robot is not None else None

    def get_initial_generation(self) -> int:
        return self.initial_generation

    def set_final_generation(self, final_generation: int):
        self.final_generation = final_generation

    def get_final_generation(self) -> int:
        return self.final_generation

    def develop(self, visualize: bool = False) -> ModularRobot:
        if self.robot is not None:
            return self.robot

        robot = self.genotype.develop(visualize)
        self.robot = robot
        return robot

    def __str__(self):
        return f"Individual(genotype={self.genotype}, fitness={self.fitness})"


def reproduce(
    parent1: Individual,
    parent2: Individual,
    rng: np.random.Generator,
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    initial_generation: int = 0,
) -> Individual:
    offspring = Genotype.crossover(parent1.genotype, parent2.genotype, rng).mutate(
        innov_db_body, innov_db_brain, rng
    )
    return Individual(
        genotype=offspring, fitness=0.0, initial_generation=initial_generation
    )
