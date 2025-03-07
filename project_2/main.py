"""Main script for the example."""

import logging

from pyrr import Vector3
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


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging()

    # Set up the random number generator.
    rng = make_rng_time_seed()
    innov_db_body = multineat.InnovationDatabase()
    innov_db_brain = multineat.InnovationDatabase()

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

    # Create the simulator.
    simulator = LocalSimulator(headless=False, num_simulators=1)

    # Simulate all scenes.
    scene_states = simulate_scenes(
        simulator=simulator,
        batch_parameters=make_standard_batch_parameters(),
        scenes=scene,
    )

    # Check if any robots are close enough to mate 
    met_before = set()  # check if robots have already met before, should reset every generational cycle
    for i in range(len(scene_states)):
        coordinates = []
        # Get all x,y,z coordinates of the robots
        for robot in robots:
            xyz = scene_states[i].get_modular_robot_simulation_state(robot).get_pose().position
            coordinates.append((xyz[0], xyz[1], xyz[2]))
        threshold = 3.0

        
        for (i, (x1, y1, z1)), (j, (x2, y2, z2)) in combinations(enumerate(coordinates), 2):
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    
        if distance <= threshold:
            pair = tuple(sorted((i, j)))  # Ensure (i, j) is always in the same order
            if pair not in met_before:  # New meeting
                met_before.add(pair)  # Mark as met
                print(f"New meeting: Robots {i} and {j} - Distance: {distance:.3f}")
                if mate_selection.mate_decision():
                    print("YAY mating")
                    offspring = mate_selection.reproduce(population[i], population[j], rng)
                    print(offspring)
                    scene.add_robot(individual, pose=Pose(Vector3([0.0, 0.0, 0.0])))
                elif not(mate_selection.mate_decision()):
                    print("Boo")
                else:
                    print("BAD")

    
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
