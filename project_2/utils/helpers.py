import math
import random
from pyrr import Vector3
from revolve2.simulators.mujoco_simulator import LocalSimulator
from torus_simulation_handler import TorusSimulationTeleportationHandler
import logging
from utils.field_limits import FieldLimits


def initialize_local_simulator(
    plane_size: float, headless: bool = False, num_simulators: int = 1
) -> LocalSimulator:
    simulator = LocalSimulator(
        viewer_type="native", headless=headless, num_simulators=num_simulators
    )
    torus_handler = TorusSimulationTeleportationHandler(plane_size=plane_size)
    simulator.register_teleport_handler(torus_handler)
    logging.info(
        f"Registered teleportation handler with plane size: {plane_size}, half size: {torus_handler.half_size}"
    )
    return simulator


def get_random_free_position(
    limits: FieldLimits, existing_positions: list[Vector3], min_dist: float = 2.0
) -> Vector3:
    """Find a random position that is not too close to existing robots."""
    while True:
        x = random.uniform(limits.get_x_min(), limits.get_x_max())
        y = random.uniform(limits.get_y_min(), limits.get_y_max())
        z = 0.0  # Assuming a flat field

        # Check distance from existing robots
        if all(
            math.sqrt((x - ex) ** 2 + (y - ey) ** 2) >= min_dist
            for ex, ey, _ in existing_positions
        ):
            return Vector3([x, y, z])  # Valid position found
