"""Brain implementation for moving a robot in a straight line."""

import numpy as np
from pyrr import Vector3

from revolve2.modular_robot import ModularRobotControlInterface
from revolve2.modular_robot.body.base import ActiveHinge
from revolve2.modular_robot.brain import Brain, BrainInstance
from revolve2.modular_robot.sensor_state import ModularRobotSensorState


class StraightLineBrainInstance(BrainInstance):
    """Instance of the StraightLineBrain, controlling a robot to move in a straight line."""

    _active_hinges: list[ActiveHinge]
    _direction: Vector3
    _amplitude: float
    _frequency: float
    _phase: float
    _hinge_patterns: list[tuple[float, float, float]]  # Amplitude, phase offset, and direction multiplier for each hinge

    def __init__(
        self,
        active_hinges: list[ActiveHinge],
        direction: Vector3 = Vector3([1.0, 0.0, 0.0]),
        amplitude: float = 0.5,
        frequency: float = 1.0,
    ):
        """
        Initialize the StraightLineBrainInstance.

        :param active_hinges: The active hinges in the robot.
        :param direction: The direction to move in (default: positive x-axis).
        :param amplitude: The amplitude of the oscillation (default: 0.5).
        :param frequency: The frequency of the oscillation (default: 1.0).
        """
        self._active_hinges = active_hinges
        self._direction = direction.normalised
        self._amplitude = amplitude
        self._frequency = frequency
        self._phase = 0.0
        
        # Initialize specific movement patterns for the gecko robot
        # This assumes the gecko robot structure with 8 active hinges (2 per leg)
        self._setup_gecko_movement_pattern()
        
    def _setup_gecko_movement_pattern(self):
        """
        Set up specific movement patterns for the gecko robot.
        
        The gecko has 4 legs with 2 hinges each:
        - Front-left leg: hinges 0-1
        - Front-right leg: hinges 2-3
        - Back-left leg: hinges 4-5
        - Back-right leg: hinges 6-7
        
        Each tuple contains: (amplitude_multiplier, phase_offset, direction_multiplier)
        """
        self._hinge_patterns = []
        num_hinges = len(self._active_hinges)
        
        if num_hinges == 8:  # Standard gecko robot
            # For a gecko robot with 8 hinges (4 legs with 2 hinges each)
            # Each tuple: (amplitude_multiplier, phase_offset, direction_multiplier)
            
            # Front-left leg
            self._hinge_patterns.append((1.0, 0.0, 1.0))          # Horizontal hinge
            self._hinge_patterns.append((0.5, np.pi/2, 1.0))      # Vertical hinge
            
            # Front-right leg
            self._hinge_patterns.append((1.0, np.pi, -1.0))        # Horizontal hinge 
            self._hinge_patterns.append((0.5, np.pi/2, 1.0))      # Vertical hinge
            
            # Back-left leg
            self._hinge_patterns.append((1.0, np.pi, 1.0))        # Horizontal hinge
            self._hinge_patterns.append((0.5, np.pi/2, 1.0))      # Vertical hinge
            
            # Back-right leg
            self._hinge_patterns.append((1.0, 0.0, -1.0))         # Horizontal hinge
            self._hinge_patterns.append((0.5, np.pi/2, 1.0))      # Vertical hinge
        else:
            # Fallback for other robot types - create a simple alternating pattern
            for i in range(num_hinges):
                phase_offset = 0.0 if i % 2 == 0 else np.pi
                self._hinge_patterns.append((1.0, phase_offset, 1.0))

    def control(
        self,
        dt: float,
        sensor_state: ModularRobotSensorState,
        control_interface: ModularRobotControlInterface,
    ) -> None:
        """
        Control the robot to move in a straight line.

        :param dt: Elapsed seconds since last call to this method.
        :param sensor_state: Sensor state at the current time step.
        :param control_interface: Interface to control the robot.
        """
        # Update the phase
        self._phase += dt * self._frequency
        
        # Apply the sinusoidal pattern to each active hinge based on the predefined patterns
        for i, active_hinge in enumerate(self._active_hinges):
            if i < len(self._hinge_patterns):
                # Get the specific pattern for this hinge
                amp_mult, phase_offset, dir_mult = self._hinge_patterns[i]
                
                # Calculate movement based on the direction vector
                if self._direction.x > 0.8:  # Moving primarily in X direction
                    movement_factor = dir_mult
                elif self._direction.y > 0.8:  # Moving primarily in Y direction
                    # Adjust for Y-direction movement
                    movement_factor = dir_mult * (0.0 if i % 2 == 0 else 1.0)
                else:
                    # For diagonal movement
                    movement_factor = dir_mult * (self._direction.x if i % 2 == 0 else self._direction.y)
                
                # Calculate the target position with the specific amplitude, phase offset and direction
                target = movement_factor * self._amplitude * amp_mult * np.sin(self._phase + phase_offset)
                
                # Set the target for this active hinge
                control_interface.set_active_hinge_target(active_hinge, target)


class StraightLineBrain(Brain):
    """Brain implementation that creates instances that control robots to move in a straight line."""

    _active_hinges: list[ActiveHinge]
    _direction: Vector3
    _amplitude: float
    _frequency: float

    def __init__(
        self,
        active_hinges: list[ActiveHinge],
        direction: Vector3 = Vector3([1.0, 0.0, 0.0]),
        amplitude: float = 0.5,
        frequency: float = 1.0,
    ):
        """
        Initialize the StraightLineBrain.

        :param active_hinges: The active hinges in the robot.
        :param direction: The direction to move in (default: positive x-axis).
        :param amplitude: The amplitude of the oscillation (default: 0.5).
        :param frequency: The frequency of the oscillation (default: 1.0).
        """
        self._active_hinges = active_hinges
        self._direction = direction
        self._amplitude = amplitude
        self._frequency = frequency

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return StraightLineBrainInstance(
            active_hinges=self._active_hinges,
            direction=self._direction,
            amplitude=self._amplitude,
            frequency=self._frequency,
        ) 