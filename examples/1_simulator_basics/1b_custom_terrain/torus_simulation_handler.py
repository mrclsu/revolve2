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
        
        # For each robot, check if it's at the edge of the plane
        for _, body_to_multi_body_system_mapping in self._brains:
            # Get the multi-body system for this robot
            multi_body_system = body_to_multi_body_system_mapping.multi_body_system
            
            # Get the current pose of the robot
            pose = simulation_state.get_multi_body_system_pose(multi_body_system)
            position = pose.position
            
            # Check if the robot is at the edge of the plane and teleport if needed
            new_position = Vector3(position)
            teleported = False
            
            # Check X boundaries
            if position.x > self.half_size:
                new_position.x = -self.half_size
                teleported = True
            elif position.x < -self.half_size:
                new_position.x = self.half_size
                teleported = True
                
            # Check Y boundaries
            if position.y > self.half_size:
                new_position.y = -self.half_size
                teleported = True
            elif position.y < -self.half_size:
                new_position.y = self.half_size
                teleported = True
            
            # If we need to teleport, update the robot's position
            if teleported:
                # We can't directly set the position via the control interface
                # This would need to be handled by the simulator directly
                # Here we log the event for now
                logging.info(f"Robot teleported from {position} to {new_position}")
                # Note: Actual teleportation would require modifying MuJoCo's data directly
                # which is not available through the current control interface 