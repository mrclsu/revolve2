from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.modular_robot_simulation import SceneSimulationState
from revolve2.standards import fitness_functions


class SimulationResult:
    def __init__(self, scene_states: SceneSimulationState):
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
                states[0].get_modular_robot_simulation_state(robot),
                states[-1].get_modular_robot_simulation_state(robot),
            )
            for robot, states in zip(robots, self.scene_states)
        ]

    def fitness(self, robots: list[ModularRobot]) -> list[float]:
        return self._xy_displacements(robots)
