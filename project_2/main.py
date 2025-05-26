"""Main script for the example."""

import logging
from uuid import UUID

from pyrr import Vector3
from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.simulation.scene.vector2.vector2 import Vector2
from genotype import Genotype
import multineat
from evaluator import Evaluator
from individual import Individual, reproduce as reproduce_individual
import config

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.standards import (
    terrains,
    mate_selection,
)
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from itertools import combinations

# from revolve2.standards.mate_selection import Reproducer
import math
import random

from incubator import Incubator
from stats import Statistics
from utils.field_limits import FieldLimits
from utils.helpers import initialize_local_simulator, get_random_free_position
from simulation_result import SimulationResult

# TODO: move to config
ITERATIONS = 300
FIELD_X_MIN, FIELD_X_MAX = -10, 10  # Adjust based on your simulation size
FIELD_Y_MIN, FIELD_Y_MAX = -10, 10

LIMITS = FieldLimits(FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    stats = Statistics(folder="stats")

    # Set up the random number generator.
    rng = make_rng_time_seed()
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # TODO: calculate from field vars
    plane_size = LIMITS.calculate_plane_size()

    # Create an initial population, with pre-trained brains
    logging.info("Generating initial population.")
    population = Incubator(
        population_size=config.POPULATION_SIZE,
        training_budget=config.INCUBATOR_TRAINING_BUDGET,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
    ).incubate()

    uuid_to_robot: dict[str, ModularRobot] = {}
    uuid_to_individual: dict[str, Individual] = {}
    for ind in population:
        robot = ind.develop(config.VISUALIZE_MAP)
        uuid_to_robot[ind.get_robot_uuid()] = robot
        uuid_to_individual[ind.get_robot_uuid()] = ind

    # Now we can create a scene and add the robots by mapping the genotypes to phenotypes
    scene = ModularRobotScene(terrain=terrains.flat(Vector2([plane_size, plane_size])))
    initial_positions: list[Vector3] = []

    logging.info("Adding initial robots to scene.")
    for robot in list(uuid_to_robot.values()):
        pos = get_random_free_position(LIMITS, initial_positions)
        scene.add_robot(robot, pose=Pose(pos))
        initial_positions.append(pos)

    simulation_results = []
    # Create the simulator.
    simulator = initialize_local_simulator(plane_size, headless=True, num_simulators=1)
    met_before = set()

    for generation in range(ITERATIONS):
        logging.info(f"Starting generation {generation}.")
        simulation_result_list = simulate_scenes(
            simulator=simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scene,
        )  # Process one scene at a time
        # TODO process each simulation state to simulate continuos mating
        simulation_result = SimulationResult(simulation_result_list)
        simulation_results.append(simulation_result)

        stats.track_individuals(uuid_to_individual)
        stats.add_generation(generation, len(population))
        stats.flush_to_json(f"generation_{generation}")

        existing_robots: list[ModularRobot] = []
        existing_robots_uuids: set[UUID] = set()
        existing_positions: list[Vector3] = []
        coordinates: list[tuple[float, float, float]] = []
        for robot, pose, _ in scene._robots:
            existing_robots.append(robot)
            existing_robots_uuids.add(robot.uuid)
            existing_positions.append(pose.position)
            xyz = (
                simulation_result.get_final_scene_state()
                .get_modular_robot_simulation_state(robot)
                .get_pose()
                .position
            )
            coordinates.append((xyz.x, xyz.y, xyz.z))

        logging.info(f"coordinates: {coordinates}")
        logging.info(f"coordinates length: {len(coordinates)}")
        logging.info(f"existing_robots length: {len(existing_robots)}")
        logging.info(f"existing_positions length: {len(existing_positions)}")

        scene = ModularRobotScene(
            terrain=terrains.flat(Vector2([plane_size, plane_size]))
        )

        # TOOD: reafactor this so it's easier to read
        for (i, (x1, y1, z1)), (j, (x2, y2, z2)) in combinations(
            enumerate(coordinates), 2
        ):
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            if distance <= config.MATING_THRESHOLD:
                r1_uuid = existing_robots[i].uuid
                r2_uuid = existing_robots[j].uuid
                pair = tuple(sorted((r1_uuid, r2_uuid)))
                if pair not in met_before:  # New meeting
                    met_before.add(pair)
                    print(f"New meeting: Robots {i} and {j} - Distance: {distance:.3f}")

                    if mate_selection.mate_decision():
                        print("YAY mating!")
                        offspring = reproduce_individual(
                            uuid_to_individual[r1_uuid],
                            uuid_to_individual[r2_uuid],
                            rng,
                            generation,
                        )
                        offspring_robot = offspring.develop(config.VISUALIZE_MAP)
                        population.append(offspring)
                        uuid_to_individual[offspring_robot.uuid] = offspring
                        uuid_to_robot[offspring_robot.uuid] = offspring_robot

        dead_individuals: list[Individual] = []
        if len(population) > config.MIN_POPULATION_SIZE:
            for ind in population:
                if (ind.initial_generation + generation) > config.MAX_AGE:
                    dead_individuals.append(ind)
                    ind.set_final_generation(generation)
                if len(population) - len(dead_individuals) < config.MIN_POPULATION_SIZE:
                    break

        for ind in dead_individuals:
            population.remove(ind)

        if len(population) > config.MAX_POPULATION_SIZE:
            population.sort(key=lambda x: x.initial_generation, reverse=True)
            population = population[: config.MAX_POPULATION_SIZE]
            removed_individuals = population[config.MAX_POPULATION_SIZE :]
            for ind in removed_individuals:
                ind.set_final_generation(generation)

        for i, coordinate in enumerate(coordinates):
            robot = existing_robots[i]
            ind = uuid_to_individual[robot.uuid]
            if ind.final_generation == -1:
                scene.add_robot(robot, pose=Pose(Vector3(coordinate)))

        for ind in population:
            if (
                ind.initial_generation == generation
                and ind.get_robot_uuid() not in existing_robots_uuids
            ):
                random_position = get_random_free_position(LIMITS, existing_positions)
                scene.add_robot(
                    ind.develop(config.VISUALIZE_MAP), pose=Pose(random_position)
                )
                existing_positions.append(random_position)


if __name__ == "__main__":
    main()
