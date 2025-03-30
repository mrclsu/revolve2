"""Torus simulation handler for wrapping the simulation environment as a torus."""

import logging
from pyrr import Vector3

from revolve2.modular_robot_simulation._modular_robot_simulation_handler import ModularRobotSimulationHandler
from revolve2.simulation.scene import ControlInterface, SimulationState


class TorusSimulationHandler(ModularRobotSimulationHandler):
    """Simulation handler that teleports robots to the other side of the plane when they reach the edge."""

    def __init__(self, plane_size: float = 1.0):
        """
        Initialize the TorusSimulationHandler.
        
        :param plane_size: The size of the plane (assumes square plane).
        """
        super().__init__()
        self.plane_size = plane_size
        self.half_size = self.plane_size / 2.0
        
    def handle(
        self,
        simulation_state: SimulationState,
        simulation_control: ControlInterface,
        dt: float,
    ) -> None:
        """
        Handle a simulation frame, checking for robots at plane edges.
        
        :param simulation_state: The current state of the simulation.
        :param simulation_control: Interface for setting control targets.
        :param dt: The time since the last call to this function.
        """
        # First let parent class handle normal brain control
        super().handle(simulation_state, simulation_control, dt)
        
    def check_teleport(self, position: Vector3) -> Vector3 | None:
        """
        Check if a robot needs to be teleported based on its position.
        
        This method is called by the teleportation system in the simulator.
        
        :param position: The current position of the robot
        :return: A new position if teleportation is needed, None otherwise
        """

        new_position = Vector3(position)

        teleported = False

        # Check X boundaries
        if position.x > self.half_size:
            logging.info(f"TorusHandler: X > {self.half_size}, teleporting to opposite side")
            new_position.x = -self.half_size
            teleported = True
        elif position.x < -self.half_size:
            logging.info(f"TorusHandler: X < -{self.half_size}, teleporting to opposite side")
            new_position.x = self.half_size
            teleported = True
            
        # Check Y boundaries
        if position.y > self.half_size:
            logging.info(f"TorusHandler: Y > {self.half_size}, teleporting to opposite side")
            new_position.y = -self.half_size
            teleported = True
        elif position.y < -self.half_size:
            logging.info(f"TorusHandler: Y < -{self.half_size}, teleporting to opposite side")
            new_position.y = self.half_size
            teleported = True

        # If we need to teleport, return the new position
        if teleported:
            logging.info(f"Teleporting robot from  x: {position.x} y: {position.y} z: {position.z} to x: {new_position.x} y: {new_position.y} z: {new_position.z}")
            return new_position

        # Otherwise, return None to indicate no teleportation needed 
        return None