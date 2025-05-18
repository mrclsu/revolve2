"""Main script for the example."""

import logging

from pyrr import Vector3
from revolve2.simulation.scene.vector2.vector2 import Vector2
from genotype import Genotype
import multineat
from evaluator import Evaluator
from individual import Individual
import config

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.standards import (
    fitness_functions,
    terrains,
    mate_selection,
)
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from itertools import combinations

# from revolve2.standards.mate_selection import Reproducer
import math
import random

from utils.field_limits import FieldLimits
from utils.helpers import initialize_local_simulator, get_random_free_position
from simulation_result import SimulationResult

# TODO: move to config
ITERATIONS = 100
FIELD_X_MIN, FIELD_X_MAX = -5, 5  # Adjust based on your simulation size
FIELD_Y_MIN, FIELD_Y_MAX = -5, 5

LIMITS = FieldLimits(FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    # Set up the random number generator.
    rng = make_rng_time_seed()
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # TODO: calculate from field vars
    plane_size = 10.0

    # Create an initial population.
    logging.info("Generating initial population.")
    initial_genotypes = [
        Genotype.random(
            innov_db_body=innov_db_body,
            innov_db_brain=innov_db_brain,
            rng=rng,
        )
        for _ in range(config.POPULATION_SIZE)
    ]

    # You can choose to not evaluate the robots if all you want is to visualize the morphologies or compute diversity to save time
    if config.EVALUATE:
        logging.info("Evaluating initial population.")
        initial_fitnesses = Evaluator(
            headless=True, num_simulators=config.NUM_SIMULATORS
        ).evaluate(initial_genotypes)
    else:
        initial_fitnesses = [
            random.uniform(0.0, 1.0) for _ in range(len(initial_genotypes))
        ]

    # Create a population of individuals, combining genotype with fitness.
    population = [
        Individual(genotype, fitness)
        for genotype, fitness in zip(initial_genotypes, initial_fitnesses, strict=True)
    ]

    # Create the robot bodies from the genotypes of the population
    # We need to map Individuals to their robot representations for position tracking
    individual_to_robot_map = {}
    for ind in population:
        robot = ind.develop(config.VISUALIZE_MAP)
        individual_to_robot_map[ind.get_robot_uuid()] = robot
    robots = list(individual_to_robot_map.values())

    # Now we can create a scene and add the robots by mapping the genotypes to phenotypes
    scene = ModularRobotScene(terrain=terrains.flat(Vector2([plane_size, plane_size])))
    i = 0
    for individual in robots:
        # By changing the value of "VISUALIZE_MAP" to true you can plot the body creation process
        scene.add_robot(individual, pose=Pose(Vector3([i, 0.0, 0.0])))
        i += 1

    simulation_results = []
    # Create the simulator.
    simulator = initialize_local_simulator(plane_size)
    # Check if any robots are close enough to mate
    # check if robots have already met before, should reset every generational cycle
    met_before = set()

    for i in range(ITERATIONS):
        simulation_result_list = simulate_scenes(
            simulator=simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scene,
        )  # Process one scene at a time
        simulation_result = SimulationResult(simulation_result_list)
        simulation_results.append(simulation_result)
        coordinates = []

        # Get positions of all robots
        robots = [robot for robot, pose, _ in scene._robots]
        for robot in robots:
            xyz = (
                simulation_result.get_final_scene_state()
                .get_modular_robot_simulation_state(robot)
                .get_pose()
                .position
            )
            coordinates.append((xyz[0], xyz[1], xyz[2]))

        # Save positions into a dictionary if needed (if not already done)
        saved_scene_state = {}
        for i, (x, y, z) in enumerate(coordinates):
            saved_scene_state[i] = {
                "position": (x, y, z),
            }

        existing_positions = [pose.position for _, pose, _ in scene._robots]

        scene = ModularRobotScene(
            terrain=terrains.flat(Vector2([plane_size, plane_size]))
        )

        # Add robots to the new scene at the saved positions (ignoring orientation)
        for i, state in saved_scene_state.items():
            position = state["position"]

            # Convert saved position to a Pose (with default orientation)
            pose = Pose(Vector3(position))  # Use default orientation (no rotation)

            # Add robot to the new scene
            scene.add_robot(robots[i], pose=pose)  # Add the robot with its new pose

        threshold = 1.2

        for (i, (x1, y1, z1)), (j, (x2, y2, z2)) in combinations(
            enumerate(coordinates), 2
        ):
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

            if distance <= threshold:
                pair = tuple(sorted((i, j)))  # Ensure consistent order
                if pair not in met_before:  # New meeting
                    met_before.add(pair)
                    print(f"New meeting: Robots {i} and {j} - Distance: {distance:.3f}")

                    if mate_selection.mate_decision():
                        print("YAY mating!")
                        offspring = mate_selection.reproduce(
                            population[i], population[j], rng
                        )
                        population.append(offspring)
                        offspring = offspring.genotype.develop(config.VISUALIZE_MAP)

                        # Add offspring dynamically
                        random_position = get_random_free_position(
                            LIMITS, existing_positions
                        )
                        scene.add_robot(offspring, pose=Pose(random_position))
                        # scene.add_robot(offspring, pose=Pose(Vector3([0.0, 0.0, 0.0])))
                        # WE SHOULD ADD THEM ON A RANDOM FREE SPOT, DEPENDS ON ENVIRONMENT

    # Calculate the xy displacements.
    # xy_displacements = [
    #     fitness_functions.xy_displacement(
    #         scene_states[0].get_modular_robot_simulation_state(robot),
    #         scene_states[-1].get_modular_robot_simulation_state(robot),
    #     )
    #     for robot in robots
    # ]

    # logging.info(xy_displacements)


if __name__ == "__main__":
    main()
