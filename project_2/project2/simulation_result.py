from enum import Enum
import logging
from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.modular_robot_simulation import SceneSimulationState
from project2.utils.fitness_functions import (
    xy_displacement,
    max_distance,
    head_stability,
)


class FitnessFunctionAlgorithm(Enum):
    XY_DISPLACEMENT = 1
    HEAD_STABILITY = 2
    MAX_DISTANCE = 3
    COMBINED = 4


class SimulationResult:
    def __init__(
        self,
        scene_states: list[SceneSimulationState],
        plane_size: float | None = None,
        movement_weight: float = 0.5,
    ):
        self.scene_states = scene_states
        self.plane_size = plane_size
        self.movement_weight = movement_weight

    def get_scene_states(self) -> list[SceneSimulationState]:
        return self.scene_states

    def get_scene_state(self, index: int) -> SceneSimulationState:
        return self.scene_states[index]

    def get_final_scene_state(self) -> SceneSimulationState:
        return self.scene_states[-1]

    def _xy_displacements(self, robots: list[ModularRobot]) -> list[float]:
        return [
            xy_displacement(
                self.scene_states[0].get_modular_robot_simulation_state(robot),
                self.scene_states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot in robots
        ]

    def _head_stability(self, robots: list[ModularRobot]) -> list[float]:
        return [head_stability(self.scene_states, robot) for robot in robots]

    def _max_distance(self, robots: list[ModularRobot]) -> list[float]:
        return [
            max_distance(self.scene_states, robot, self.plane_size) for robot in robots
        ]

    def _combined(self, robots: list[ModularRobot]) -> list[float]:
        head_stability_scores = self._head_stability(robots)
        max_distance_scores = self._max_distance(robots)

        return [
            head_stability_score * (1 - self.movement_weight)
            + max_distance_score * self.movement_weight
            for head_stability_score, max_distance_score in zip(
                head_stability_scores, max_distance_scores
            )
        ]

    def get_all_fitness_metrics(
        self, robots: list[ModularRobot]
    ) -> dict[str, list[float]]:
        """Calculate and return both MAX_DISTANCE and HEAD_STABILITY fitness metrics."""
        return {
            "MAX_DISTANCE": self._max_distance(robots),
            "HEAD_STABILITY": self._head_stability(robots),
        }

    def fitness(
        self,
        robots: list[ModularRobot],
        fitness_function: FitnessFunctionAlgorithm = FitnessFunctionAlgorithm.XY_DISPLACEMENT,
    ) -> list[float]:
        if fitness_function == FitnessFunctionAlgorithm.XY_DISPLACEMENT:
            return self._xy_displacements(robots)
        elif fitness_function == FitnessFunctionAlgorithm.HEAD_STABILITY:
            return self._head_stability(robots)
        elif fitness_function == FitnessFunctionAlgorithm.MAX_DISTANCE:
            return self._max_distance(robots)
        elif fitness_function == FitnessFunctionAlgorithm.COMBINED:
            return self._combined(robots)
        else:
            raise ValueError(f"Invalid fitness function: {fitness_function}")
