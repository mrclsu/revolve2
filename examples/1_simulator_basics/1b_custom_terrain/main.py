"""Main script for the example."""

import math

from pyrr import Quaternion, Vector3

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from revolve2.simulation.scene import AABB, Color, Pose
from revolve2.simulation.scene.geometry import GeometryBox, GeometryPlane, GeometrySphere
from revolve2.simulation.scene.geometry.textures import MapType
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.simulators.mujoco_simulator.textures import Checker, Flat, Gradient
from revolve2.standards.modular_robots_v2 import gecko_v2, snake_v2
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
import logging

# Import our custom classes
from torus_simulation_handler import TorusSimulationHandler
from straight_line_brain import StraightLineBrain


def make_custom_terrain(plane_size: float = 1.0) -> Terrain:
    """
    Create a custom terrain.

    :returns: The created terrain.
    """
    # A terrain is a collection of static geometries.
    # Here we create a simple terrain uses some boxes.
    return Terrain(
        static_geometry=[
            GeometryPlane(
                pose=Pose(position=Vector3([0.0, 0.0, 0.0]), orientation=Quaternion()),
                mass=0.0,
                size=Vector3([plane_size, plane_size, 1.0]),
                texture=Checker(
                    primary_color=Color(170, 170, 180, 255),
                    secondary_color=Color(150, 150, 150, 255),
                    map_type=MapType.CUBE,
                ),
            ),
        ]
    )


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    # Set up the random number generator.
    rng = make_rng_time_seed()

    # Create a robot
    body = snake_v2()
    
    # Find all active hinges in the robot body
    active_hinges = body.find_modules_of_type(ActiveHinge)
    
    # Create the StraightLineBrain for the robot
    # Here we're setting the direction to move along the x-axis (forward)
    brain = StraightLineBrain(
        active_hinges=active_hinges,
        direction=Vector3([1.0, 0.0, 0.0]),  # Move along the positive x-axis
        amplitude=0.7,  # Amplitude of the oscillation - increased for more pronounced movement
        frequency=1.0,  # Frequency of the oscillation (Hz) - slightly reduced for stability
    )
    
    robot = ModularRobot(body, brain)

    # Define the plane size for our torus world
    plane_size = 2.0  # Size of the plane (side length)
    

    
    # Create the scene with our custom torus handler
    scene = ModularRobotScene(terrain=make_custom_terrain(plane_size))
    scene.add_robot(robot)

    batch_parameters = make_standard_batch_parameters()
    batch_parameters.simulation_time = 1200000  # Here we update our simulation time.

    # Create the simulator and register our teleportation handler
    simulator = LocalSimulator(viewer_type="native")

    # Create a torus handler to manage teleportation
    torus_handler = TorusSimulationHandler(plane_size=plane_size)
    # Register our teleportation handler function - this will check and handle teleportation
    simulator.register_teleport_handler(torus_handler.check_teleport)
    logging.info(f"Registered teleportation handler with plane size: {plane_size}, half size: {torus_handler.half_size}")
    
    # Simulate the scene with active teleportation
    simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene,
    )


if __name__ == "__main__":
    main()
