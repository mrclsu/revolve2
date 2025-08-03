"""Main script for the example."""

import logging
from project2.config import Config
from uuid import UUID

from pyrr import Vector3
from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.simulation.scene.vector2.vector2 import Vector2

import multineat

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed, make_rng_time_seed2
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from itertools import combinations

# from revolve2.standards.mate_selection import Reproducer
import math
# import random


from project2.individual import Individual, reproduce as reproduce_individual
from project2.incubator import Incubator
from project2.utils.helpers import initialize_local_simulator, get_random_free_position
from project2.simulation_result import SimulationResult
from project2.stats import Statistics
import project2.mate_selection as mate_selection
from project2.death_mechanism import apply_death_mechanism


def main(config: Config, folder_name: str = "stats") -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    stats = Statistics(folder_name=folder_name)

    # Set up the random number generator.
    rng, seed = make_rng_time_seed2()
    stats.save_seed(seed)
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    plane_size = config.LIMITS.calculate_plane_size()

    # Create an initial population, with pre-trained brains
    logging.info("Generating initial population.")
    population = Incubator(
        population_size=config.POPULATION_SIZE,
        training_budget=config.INCUBATOR_TRAINING_BUDGET,
        innov_db_body=innov_db_body,
        innov_db_brain=innov_db_brain,
        rng=rng,
        num_simulators=config.NUM_SIMULATORS,
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
        pos = get_random_free_position(config.LIMITS, initial_positions)
        scene.add_robot(robot, pose=Pose(pos))
        initial_positions.append(pos)

    simulation_results = []
    # Create the simulator.
    simulator = initialize_local_simulator(
        plane_size, headless=config.SIMULATION_HEADLESS, num_simulators=1
    )
    met_before: dict[tuple, int] = {}

    for generation in range(config.ITERATIONS):
        logging.info(f"Starting generation {generation}.")
        simulation_result_list = simulate_scenes(
            simulator=simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scene,
        )

        simulation_result = SimulationResult(
            simulation_result_list,
            plane_size=plane_size,
            movement_weight=config.MOVEMENT_WEIGHT,
        )
        simulation_results.append(simulation_result)

        current_robots = [robot for robot, _, _ in scene._robots]
        fitness_values = simulation_result.fitness(
            current_robots, config.FITNESS_FUNCTION_ALGORITHM
        )

        # Get additional fitness metrics for stats tracking
        all_fitness_metrics = simulation_result.get_all_fitness_metrics(current_robots)

        # Map fitness metrics to robot UUIDs
        fitness_metrics_by_uuid = {}
        for metric_name, metric_values in all_fitness_metrics.items():
            fitness_metrics_by_uuid[metric_name] = {}
            for robot, metric_value in zip(current_robots, metric_values):
                if robot.uuid in uuid_to_individual:
                    fitness_metrics_by_uuid[metric_name][robot.uuid] = metric_value

        for robot, fitness in zip(current_robots, fitness_values):
            if robot.uuid in uuid_to_individual:
                uuid_to_individual[robot.uuid].fitness = fitness

        stats.track_individuals(uuid_to_individual, generation, fitness_metrics_by_uuid)
        stats.add_generation(generation, len(population))
        stats.flush_to_json(f"generation_{generation}")

        existing_robots: list[ModularRobot] = []
        existing_robots_uuids: set[UUID] = set()
        existing_positions: list[Vector3] = []
        coordinates: list[tuple[float, float, float]] = []
        final_coordinates: list[tuple[float, float, float]] = []
        for robot, pose, _ in scene._robots:
            existing_robots.append(robot)
            existing_robots_uuids.add(robot.uuid)
            existing_positions.append(pose.position)
            states = simulation_result.get_scene_states()
            state_id = 0
            for state in states:
                state_id += 1
                xyz = (
                    state.get_modular_robot_simulation_state(robot).get_pose().position
                )
                coordinates.append((xyz.x, xyz.y, xyz.z, robot, state_id))
            final_state = simulation_result.get_final_scene_state()
            final_xyz = (
                final_state.get_modular_robot_simulation_state(robot)
                .get_pose()
                .position
            )
            final_coordinates.append((final_xyz.x, final_xyz.y, final_xyz.z))

        logging.info(f"coordinates length: {len(coordinates)}")
        logging.info(f"existing_robots length: {len(existing_robots)}")
        logging.info(f"existing_positions length: {len(existing_positions)}")

        scene = ModularRobotScene(
            terrain=terrains.flat(Vector2([plane_size, plane_size]))
        )

        # TOOD: reafactor this so it's easier to read
        for (i, (x1, y1, z1, robot1, state_id1)), (
            j,
            (x2, y2, z2, robot2, state_id2),
        ) in combinations(enumerate(coordinates), 2):
            r1_uuid = robot1.uuid
            r2_uuid = robot2.uuid
            if r1_uuid != r2_uuid and state_id1 == state_id2:
                distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
                if distance <= config.MATING_THRESHOLD:
                    pair = tuple(sorted((r1_uuid, r2_uuid)))
                    if (
                        pair not in met_before
                        or generation - met_before[pair] >= config.MATING_COOLDOWN
                    ):
                        met_before[pair] = generation
                        logging.info(
                            f"Meeting: Robots {i} and {j} - Distance: {distance:.3f}"
                        )
                        stats.increment_meeting_count(r1_uuid)
                        stats.increment_meeting_count(r2_uuid)

                        if mate_selection.mate_decision(
                            config.MATE_SELECTION_STRATEGY,
                            uuid_to_individual[r1_uuid],
                            uuid_to_individual[r2_uuid],
                            population,
                            config.MATE_SELECTION_THRESHOLD,
                        ):
                            logging.info("YAY mating!")

                            # Increment offspring count for both parents
                            stats.increment_offspring_count(r1_uuid)
                            stats.increment_offspring_count(r2_uuid)

                            offspring = reproduce_individual(
                                uuid_to_individual[r1_uuid],
                                uuid_to_individual[r2_uuid],
                                rng,
                                innov_db_body,
                                innov_db_brain,
                                generation,
                            )
                            offspring_robot = offspring.develop(config.VISUALIZE_MAP)
                            population.append(offspring)
                            uuid_to_individual[offspring_robot.uuid] = offspring
                            uuid_to_robot[offspring_robot.uuid] = offspring_robot

        # Apply death mechanism based on configuration
        dead_individuals = apply_death_mechanism(
            population=population,
            current_generation=generation,
            death_mechanism=config.DEATH_MECHANISM,
            max_population_size=config.MAX_POPULATION_SIZE,
            min_population_size=config.MIN_POPULATION_SIZE,
            max_age=config.MAX_AGE,
        )

        for ind in dead_individuals:
            ind.set_final_generation(generation)
            population.remove(ind)

        for i, coordinate in enumerate(final_coordinates):
            robot = existing_robots[i]
            ind = uuid_to_individual[robot.uuid]
            # Reset teleport coordinates for the next generation
            robot.teleport_coordinates = []
            if ind.final_generation == -1:
                scene.add_robot(robot, pose=Pose(Vector3(coordinate)))

        for ind in population:
            if (
                ind.initial_generation == generation
                and ind.get_robot_uuid() not in existing_robots_uuids
            ):
                random_position = get_random_free_position(
                    config.LIMITS, existing_positions
                )
                scene.add_robot(
                    ind.develop(config.VISUALIZE_MAP), pose=Pose(random_position)
                )
                existing_positions.append(random_position)


if __name__ == "__main__":
    import argparse
    import os
    import glob

    # Dynamically get available config choices from configs/ folder
    config_dir = os.path.join(os.path.dirname(__file__), "project2", "configs")
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    available_configs = [os.path.splitext(os.path.basename(f))[0] for f in config_files]
    available_configs.sort()  # Sort for consistent ordering

    parser = argparse.ArgumentParser(
        description="Run simulation with different configs"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=available_configs,
        default="config1"
        if "config1" in available_configs
        else (available_configs[0] if available_configs else "config1"),
        help=f"Config name to use. Available configs: {', '.join(available_configs)}",
    )
    args = parser.parse_args()

    # Create a Config instance with the specified config name
    config = Config(args.config)

    main(config, f"stats/{args.config}")
