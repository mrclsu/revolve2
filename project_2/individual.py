"""Individual class."""

from dataclasses import dataclass

from revolve2.modular_robot._modular_robot import ModularRobot

from genotype import Genotype


@dataclass
class Individual:
    """An individual in a population."""

    genotype: Genotype
    fitness: float

    robot_uuid: str | None = None

    def __init__(self, genotype: Genotype, fitness: float):
        self.genotype = genotype
        self.fitness = fitness

    def set_robot_uuid(self, robot_uuid: str):
        self.robot_uuid = robot_uuid

    def get_robot_uuid(self) -> str | None:
        return self.robot_uuid

    def develop(self, visualize: bool = False) -> ModularRobot:
        robot = self.genotype.develop(visualize)
        self.set_robot_uuid(robot.uuid)
        return robot

    def __str__(self):
        return f"Individual(genotype={self.genotype}, fitness={self.fitness})"
