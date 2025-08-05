"""Main script for the example."""

from datetime import datetime
import json
import logging
from time import perf_counter

from revolve2.modular_robot.brain.cpg._brain_cpg_network_neighbor_random import (
    BrainCpgNetworkNeighborRandom,
)

from revolve2.modular_robot_simulation._to_batch import to_batch
from revolve2.standards.modular_robots_v2 import spider_v2, snake_v2, ant_v2, gecko_v2
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from revolve2.modular_robot._modular_robot import ModularRobot
from revolve2.simulation.scene.vector2.vector2 import Vector2

from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import (
    make_rng_time_seed2,
)
from revolve2.modular_robot_simulation import ModularRobotScene
from revolve2.simulation.scene import Pose
from revolve2.standards import terrains


import numpy as np

from project2.utils.helpers import get_random_free_position, initialize_local_simulator
from project2.utils.field_limits import FieldLimits


def make_spider_robots(amount: int, rng: np.random.Generator) -> list[ModularRobot]:
    robots = []
    for _ in range(amount):
        body = spider_v2()
        brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
        robots.append(ModularRobot(body, brain))
    return robots


def make_random_robots(amount: int, rng: np.random.Generator) -> list[ModularRobot]:
    bodies = [spider_v2(), snake_v2(), ant_v2(), gecko_v2()]
    robots = []
    for _ in range(amount):
        body = rng.choice(bodies)
        brain = BrainCpgNetworkNeighborRandom(body=body, rng=rng)
        robots.append(ModularRobot(body, brain))
    return robots


def time_experiment(
    robot_count: int,
    robot_type: str,
    rng: np.random.Generator,
    limits: FieldLimits,
    iterations: int = 10,
) -> list[float]:
    results = []
    for i in range(iterations):
        print(f"Starting iteration {i}")
        if robot_type == "spider":
            robots = make_spider_robots(robot_count, rng)
        elif robot_type == "random":
            robots = make_random_robots(robot_count, rng)
        else:
            raise ValueError(f"Invalid robot type: {robot_type}")

        simulator = initialize_local_simulator(
            limits.calculate_plane_size(), headless=True, num_simulators=1
        )

        parameters = make_standard_batch_parameters(simulation_time=5)

        existing_positions = []
        scene = ModularRobotScene(
            terrain=terrains.flat(
                Vector2([limits.calculate_plane_size(), limits.calculate_plane_size()])
            )
        )
        for robot in robots:
            pos = get_random_free_position(limits, existing_positions)
            scene.add_robot(robot, pose=Pose(pos))
            existing_positions.append(pos)

        batch, _ = to_batch(scene, parameters)

        start_time = perf_counter()
        simulator.simulate_batch(batch)
        end_time = perf_counter()
        results.append(end_time - start_time)
        print(f"Iteration {i} Simulation time: {end_time - start_time} seconds")

    return results


def main() -> None:
    """Run the simulation."""
    # Set up logging.
    setup_logging(level=logging.WARNING)
    limits = FieldLimits(-5, 5, -5, 5)
    iterations = 10
    robot_counts = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    robot_types = ["spider", "random"]

    # Set up the random number generator.
    rng, seed = make_rng_time_seed2()
    print(f"Seed: {seed}")

    output_file = (
        f"run_time_results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
    )

    with open(output_file, "w") as f:
        for robot_type in robot_types:
            for robot_count in robot_counts:
                results = time_experiment(
                    robot_count, robot_type, rng, limits, iterations
                )
                print(f"Results for {robot_count} {robot_type}: {results}")
                result_obj = {
                    "robot_count": robot_count,
                    "robot_type": robot_type,
                    "results": results,
                    "seed": seed,
                }
                f.write(json.dumps(result_obj) + "\n")
                f.flush()


if __name__ == "__main__":
    main()
