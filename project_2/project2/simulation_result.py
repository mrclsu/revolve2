from enum import Enum
from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.modular_robot_simulation import SceneSimulationState
from revolve2.standards import fitness_functions


class FitnessFunctionAlgorithm(Enum):
    XY_DISPLACEMENT = 1
    HEAD_STABILITY = 2


class SimulationResult:
    def __init__(self, scene_states: list[SceneSimulationState]):
        self.scene_states = scene_states

    def get_scene_states(self) -> list[SceneSimulationState]:
        return self.scene_states

    def get_scene_state(self, index: int) -> SceneSimulationState:
        return self.scene_states[index]

    def get_final_scene_state(self) -> SceneSimulationState:
        return self.scene_states[-1]

    def _xy_displacements(self, robots: list[ModularRobot]) -> list[float]:
        return [
            fitness_functions.xy_displacement(
                self.scene_states[0].get_modular_robot_simulation_state(robot),
                self.scene_states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot in robots
        ]

    def fitness(
        self,
        robots: list[ModularRobot],
        fitness_function: FitnessFunctionAlgorithm = FitnessFunctionAlgorithm.XY_DISPLACEMENT,
    ) -> list[float]:
        if fitness_function == FitnessFunctionAlgorithm.XY_DISPLACEMENT:
            return self._xy_displacements(robots)
        elif fitness_function == FitnessFunctionAlgorithm.HEAD_STABILITY:
            raise NotImplementedError("Head stability fitness function not implemented")
        else:
            raise ValueError(f"Invalid fitness function: {self.fitness_function}")
