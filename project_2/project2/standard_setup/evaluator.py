from project2.genotype import Genotype


from revolve2.experimentation.evolution.abstract_elements import Evaluator as Eval
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from revolve2.modular_robot_simulation import (
    ModularRobotScene,
    Terrain,
    simulate_scenes,
)
from project2.simulation_result import (
    FitnessFunctionAlgorithm,
    SimulationResult,
)
from project2.utils.helpers import initialize_local_simulator


class Evaluator(Eval):
    """Provides evaluation of robots."""

    _simulator: LocalSimulator
    _terrain: Terrain

    def __init__(
        self,
        headless: bool,
        num_simulators: int,
        plane_size: float,
        fitness_function_algorithm: FitnessFunctionAlgorithm,
        movement_weight: float,
    ) -> None:
        """
        Initialize this object.

        :param headless: `headless` parameter for the physics simulator.
        :param num_simulators: `num_simulators` parameter for the physics simulator.
        """
        self._simulator = initialize_local_simulator(
            headless=headless, num_simulators=num_simulators, plane_size=plane_size
        )
        self._terrain = terrains.flat()
        self._fitness_function_algorithm = fitness_function_algorithm
        self._plane_size = plane_size
        self._movement_weight = movement_weight

    def evaluate(
        self, population: list[Genotype]
    ) -> tuple[list[float], list[dict[str, float]]]:
        """
        Evaluate multiple robots.

        Fitness is the distance traveled on the xy plane.

        :param population: The robots to simulate.
        :returns: Fitnesses of the robots.
        """
        robots = [genotype.develop() for genotype in population]
        scenes = []
        for robot in robots:
            scene = ModularRobotScene(terrain=self._terrain)
            scene.add_robot(robot)
            scenes.append(scene)

        # Simulate all scenes.
        scene_states = simulate_scenes(
            simulator=self._simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scenes,
        )

        simulation_results = [
            SimulationResult(
                scene_state,
                plane_size=self._plane_size,
                movement_weight=self._movement_weight,
            )
            for scene_state in scene_states
        ]

        fitness_values = [
            results.fitness([robot], self._fitness_function_algorithm)
            for robot, results in zip(robots, simulation_results)
        ]

        all_fitness_metrics = [
            results.get_all_fitness_metrics([robot])
            for robot, results in zip(robots, simulation_results)
        ]

        return fitness_values, all_fitness_metrics
