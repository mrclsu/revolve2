"""Main script for the example."""

import logging

from pyrr import Vector3
from revolve2.simulation.scene.vector2.vector2 import Vector2
from genotype import Genotype
import multineat
from evaluator import Evaluator
from individual import Individual
import config
import numpy as np

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulation.scene import Pose
from revolve2.simulators.mujoco_simulator import LocalSimulator
from revolve2.standards import fitness_functions, modular_robots_v2, terrains, mate_selection
from revolve2.standards.simulation_parameters import make_standard_batch_parameters
from itertools import combinations
#from revolve2.standards.mate_selection import Reproducer
import math
import random

from torus_simulation_handler import TorusSimulationTeleportationHandler


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    # Set up the random number generator.
    rng = make_rng_time_seed()
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

    # scene size 
    FIELD_X_MIN, FIELD_X_MAX = -5, 5  # Adjust based on your simulation size
    FIELD_Y_MIN, FIELD_Y_MAX = -5, 5

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
    robots = [
        individual.genotype.develop(config.VISUALIZE_MAP) for individual in population
    ]


    # Create the robots.
    #bodies = [
    #    modular_robots_v2.gecko_v2(),
    #    modular_robots_v2.ant_v2(),
    #    modular_robots_v2.snake_v2(),
    #    modular_robots_v2.spider_v2(),
    #]
    #brains = [BrainCpgNetworkNeighborRandom(body, rng) for body in bodies]
    #robots = [ModularRobot(body, brain) for body, brain in zip(bodies, brains)]


    # Now we can create a scene and add the robots by mapping the genotypes to phenotypes
    scene = ModularRobotScene(terrain=terrains.flat())
    i = 0
    for individual in robots:
        # By changing the value of "VISUALIZE_MAP" to true you can plot the body creation process
        scene.add_robot(individual, pose=Pose(Vector3([i, 0.0, 0.0])))
        i += 1
    """
    Contrary to the previous examples, we now create a single scene and put all robots in it. 
    We place the robots at separate locations in the terrain so they do not overlap at the start of the simulation.
    """
    #scene = ModularRobotScene(terrain=terrains.flat())
    #poses = [
    #    Pose(Vector3([1.0, 0.0, 0.0])),
    #    Pose(Vector3([-1.0, 0.0, 0.0])),
    #    Pose(Vector3([0.0, 1.0, 0.0])),
    #    Pose(Vector3([0.0, -1.0, 0.0])),
    #]
    #for robot, pose in zip(robots, poses):
    #    scene.add_robot(robot, pose=pose)

    def get_random_free_position(existing_positions, min_dist=2.0):
        """Find a random position that is not too close to existing robots."""
        while True:
            x = random.uniform(FIELD_X_MIN, FIELD_X_MAX)
            y = random.uniform(FIELD_Y_MIN, FIELD_Y_MAX)
            z = 0.0  # Assuming a flat field

            # Check distance from existing robots
            if all(math.sqrt((x - ex) ** 2 + (y - ey) ** 2) >= min_dist for ex, ey, _ in existing_positions):
                return Vector3([x, y, z])  # Valid position found

    # Create the simulator.
    simulator = LocalSimulator(viewer_type='native', headless=False, num_simulators=1)
    # Check if any robots are close enough to mate 
    met_before = set()  # check if robots have already met before, should reset every generational cycle
    for i in range(100):  # Loop for iterative simulation steps (or replace with a fixed number of iterations)
        scene_state = simulate_scenes(
            simulator=simulator,
            batch_parameters=make_standard_batch_parameters(),
            scenes=scene,
        )[0]  # Process one scene at a time

        coordinates = []

        # Get positions of all robots
        robots = [robot for robot, pose, _ in scene._robots]
        for robot in robots:
            xyz = scene_state.get_modular_robot_simulation_state(robot).get_pose().position
            coordinates.append((xyz[0], xyz[1], xyz[2]))

        # Save positions into a dictionary if needed (if not already done)
        saved_scene_state = {}
        for i, (x, y, z) in enumerate(coordinates):
            saved_scene_state[i] = {
                "position": (x, y, z),
            }
        
        existing_positions = [pose.position for _, pose, _ in scene._robots]

        plane_size = 10.0 # TODO USE THE CONSTANTS TO CALCULATE THIS
        scene = ModularRobotScene(terrain=terrains.flat(Vector2([plane_size, plane_size])))
        torus_handler = TorusSimulationTeleportationHandler(plane_size=plane_size)
        simulator.register_teleport_handler(torus_handler)
        logging.info(f"Registered teleportation handler with plane size: {plane_size}, half size: {torus_handler.half_size}")


        # Add robots to the new scene at the saved positions (ignoring orientation)
        for i, state in saved_scene_state.items():
            position = state["position"]

            # Convert saved position to a Pose (with default orientation)
            pose = Pose(Vector3(position))  # Use default orientation (no rotation)

            # Add robot to the new scene
            scene.add_robot(robots[i], pose=pose)  # Add the robot with its new pose

        threshold = 1.2

        for (i, (x1, y1, z1)), (j, (x2, y2, z2)) in combinations(enumerate(coordinates), 2):
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

            if distance <= threshold:
                pair = tuple(sorted((i, j)))  # Ensure consistent order
                if pair not in met_before:  # New meeting
                    met_before.add(pair)
                    print(f"New meeting: Robots {i} and {j} - Distance: {distance:.3f}")

                    if mate_selection.mate_decision():
                        print("YAY mating!")
                        offspring = mate_selection.reproduce(population[i], population[j], rng)
                        population.append(offspring)
                        offspring = offspring.genotype.develop(config.VISUALIZE_MAP)
                        

                        # Add offspring dynamically
                        random_position = get_random_free_position(existing_positions)
                        scene.add_robot(offspring, pose=Pose(random_position))
                        #scene.add_robot(offspring, pose=Pose(Vector3([0.0, 0.0, 0.0])))
                        # WE SHOULD ADD THEM ON A RANDOM FREE SPOT, DEPENDS ON ENVIRONMENT
        
        
    
    # Calculate the xy displacements.
    xy_displacements = [
        fitness_functions.xy_displacement(
            scene_states[0].get_modular_robot_simulation_state(robot),
            scene_states[-1].get_modular_robot_simulation_state(robot),
        )
        for robot in robots
    ]

    logging.info(xy_displacements)


if __name__ == "__main__":
    main()
