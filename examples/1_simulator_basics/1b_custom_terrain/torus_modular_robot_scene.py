"""Torus modular robot scene for wrapping the simulation environment as a torus."""

from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot_simulation import ModularRobotScene, Terrain
from revolve2.simulation.scene import MultiBodySystem, Scene, UUIDKey
from revolve2.modular_robot_simulation._build_multi_body_systems import BodyToMultiBodySystemConverter
from revolve2.modular_robot_simulation._convert_terrain import convert_terrain

from torus_simulation_handler import TorusSimulationHandler


class TorusModularRobotScene(ModularRobotScene):
    """A custom scene that uses our torus simulation handler."""
    
    def __init__(self, terrain: Terrain, plane_size: float = 1.0):
        """
        Initialize the torus robot scene.
        
        :param terrain: The terrain for the scene.
        :param plane_size: The size of the plane (assumes square plane).
        """
        super().__init__(terrain)
        self.plane_size = plane_size
        
    def to_simulation_scene(
        self,
    ) -> tuple[Scene, dict[UUIDKey[ModularRobot], MultiBodySystem]]:
        """
        Convert this to a simulation scene with our custom handler.

        :returns: The created scene with a torus handler.
        """
        # Use our custom handler instead of the default one
        handler = TorusSimulationHandler(plane_size=self.plane_size)
        scene = Scene(handler=handler)
        modular_robot_to_multi_body_system_mapping: dict[
            UUIDKey[ModularRobot], MultiBodySystem
        ] = {}

        # Add terrain
        scene.add_multi_body_system(convert_terrain(self.terrain))

        # Add robots
        converter = BodyToMultiBodySystemConverter()
        for robot, pose, translate_z_aabb in self._robots:
            # Convert all bodies to multi body systems and add them to the simulation scene
            (
                multi_body_system,
                body_to_multi_body_system_mapping,
            ) = converter.convert_robot_body(
                body=robot.body, pose=pose, translate_z_aabb=translate_z_aabb
            )
            scene.add_multi_body_system(multi_body_system)
            handler.add_robot(
                robot.brain.make_instance(), body_to_multi_body_system_mapping
            )
            modular_robot_to_multi_body_system_mapping[UUIDKey(robot)] = (
                multi_body_system
            )

        for interactive_object in self._interactive_objects:
            scene.add_multi_body_system(interactive_object)

        return scene, modular_robot_to_multi_body_system_mapping 