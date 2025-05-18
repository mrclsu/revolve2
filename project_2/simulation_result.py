from revolve2.modular_robot_simulation import SceneSimulationState


class SimulationResult:
    def __init__(self, scene_states: SceneSimulationState):
        self.scene_states = scene_states

    def get_scene_states(self) -> list[SceneSimulationState]:
        return self.scene_states

    def get_scene_state(self, index: int) -> SceneSimulationState:
        return self.scene_states[index]

    def get_final_scene_state(self) -> SceneSimulationState:
        return self.scene_states[-1]
