from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.modular_robot_simulation._modular_robot_simulation_state import (
    ModularRobotSimulationState,
)
from revolve2.modular_robot_simulation._scene_simulation_state import (
    SceneSimulationState,
)
from revolve2.simulation.scene._pose import Pose
from revolve2.standards.fitness_functions import (
    xy_displacement as xy_displacement_standard,
)
import math
import numpy as np
import logging


def xy_displacement(
    begin_state: ModularRobotSimulationState, end_state: ModularRobotSimulationState
) -> float:
    return xy_displacement_standard(begin_state, end_state)


def torus_aware_xy_displacement(
    begin_state: ModularRobotSimulationState,
    end_state: ModularRobotSimulationState,
    robot: ModularRobot,
) -> float:
    if len(robot.teleport_coordinates) > 0:
        start_pos = begin_state.get_pose().position
        end_pos = end_state.get_pose().position

        teleported = (
            abs(start_pos.x - end_pos.x) > 1 or abs(start_pos.y - end_pos.y) > 1
        ) and any(
            [
                (
                    abs(start_pos.x - teleport_coord[0].x) < 1
                    and abs(start_pos.y - teleport_coord[0].y) < 1
                    and abs(end_pos.x - teleport_coord[1].x) < 1
                    and abs(end_pos.y - teleport_coord[1].y) < 1
                )
                for teleport_coord in robot.teleport_coordinates
            ]
        )
        if teleported:
            logging.info(
                f"Teleported in current state: {start_pos}, {end_pos}, {abs(start_pos.x - end_pos.x) > 1}, {abs(start_pos.y - end_pos.y) > 1}"
            )
            return 0.0

    # If no teleportation occurred, use regular distance calculation
    return xy_displacement(begin_state, end_state)


def head_stability(
    scene_states: list[SceneSimulationState],
    robot: ModularRobot,
) -> float:
    """
    Calculate core module (head) stability on a 0-1 scale.

    This function measures how consistent the robot's core module orientation is
    throughout the simulation. The core module serves as the "head" of the modular robot
    and is the central component that all other modules (joints, bricks) are attached to.

    A value of 1.0 means perfectly stable core orientation (no orientation change),
    while 0.0 means highly unstable core orientation (large orientation changes).

    This function now attempts to track the actual core module pose when available,
    falling back to overall robot pose if core module tracking is not implemented.

    :param scene_states: List of simulation states throughout the simulation
    :param robot: The modular robot to track (contains core module as robot.body.core)
    :return: Core stability score (0.0 to 1.0)
    """
    if len(scene_states) < 2:
        return 1.0  # No change possible with less than 2 states

    # Get all core module poses throughout the simulation
    poses: list[Pose] = []

    for state in scene_states:
        robot_state = state.get_modular_robot_simulation_state(robot)

        # Try to get the core module pose directly
        try:
            if hasattr(robot_state, "get_core_absolute_pose"):
                core_pose = robot_state.get_core_absolute_pose(robot.body.core)
                poses.append(core_pose)
            else:
                # Fall back to robot pose
                logging.warning("Core tracking failed, falling back to robot pose")
                poses.append(robot_state.get_pose())
        except (AttributeError, KeyError, NotImplementedError):
            # Fall back to robot pose if core tracking fails
            logging.warning("Core tracking failed, falling back to robot pose")
            poses.append(robot_state.get_pose())

    angular_deviations = []

    for i in range(len(poses) - 1):
        current_orientation = poses[i].orientation
        next_orientation = poses[i + 1].orientation

        # Calculate the angular distance between two quaternions
        # Using the formula: angle = 2 * arccos(|dot_product|)
        # Clamp the dot product to avoid numerical issues
        dot_product = abs(np.dot(current_orientation, next_orientation))
        dot_product = min(1.0, max(-1.0, dot_product))

        # Calculate angular deviation in radians
        angular_deviation = 2 * math.acos(dot_product)
        angular_deviations.append(angular_deviation)

    if not angular_deviations:
        return 1.0  # Perfect stability if no deviations

    # Calculate the average angular deviation
    avg_angular_deviation = sum(angular_deviations) / len(angular_deviations)

    # Convert to stability score (0-1 scale)
    # Use an exponential decay function to map angular deviation to stability
    # Higher deviations result in lower stability scores
    # The constant 2.0 controls the sensitivity (can be adjusted)
    stability_score = math.exp(-avg_angular_deviation * 2.0)

    return max(0.0, min(1.0, stability_score))


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
            logging.warning("Plane size is not set, using regular distance calculation")
            # Use regular distance calculation
            total_distance += xy_displacement(current_robot_state, next_robot_state)

    # Clamp the distance to the plane size, so that the fitness function is always between 0 and 1
    if plane_size is not None:
        return min(1.0, total_distance / plane_size)
    else:
        logging.warning("Plane size is not set, using regular distance calculation")
        return total_distance
