from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.modular_robot_simulation._modular_robot_simulation_state import (
    ModularRobotSimulationState,
)
from revolve2.modular_robot_simulation._scene_simulation_state import (
    SceneSimulationState,
)
from revolve2.standards.fitness_functions import (
    xy_displacement as xy_displacement_standard,
)

import math


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    return xy_displacement_standard(begin_state, end_state)


def torus_aware_xy_displacement(
    begin_state: ModularRobotSimulationState,
    end_state: ModularRobotSimulationState,
    robot: ModularRobot,
) -> float:
    if robot.has_teleported:
        return 0.0

    # If no teleportation occurred, use regular distance calculation
    return xy_displacement(begin_state, end_state)


def max_distance(
    scene_states: list[SceneSimulationState],
    robot: ModularRobot,
    plane_size: float | None = None,
) -> float:
    """
    Calculate maximum distance traveled by a robot.

    :param scene_states: List of simulation states
    :param robot: The robot to track
    :param plane_size: Size of torus plane (if None, uses regular distance calculation)
    :return: Total distance traveled
    """
    total_distance = 0
    for i in range(len(scene_states) - 1):
        current_robot_state = scene_states[i].get_modular_robot_simulation_state(robot)
        next_robot_state = scene_states[i + 1].get_modular_robot_simulation_state(robot)

        if plane_size is not None:
            # Use torus-aware distance calculation
            total_distance += torus_aware_xy_displacement(
                current_robot_state, next_robot_state, robot
            )
        else:
            # Use regular distance calculation
            total_distance += xy_displacement(current_robot_state, next_robot_state)

    return total_distance
