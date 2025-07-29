import pickle
import codecs
import json

from revolve2.modular_robot_simulation._modular_robot_scene import ModularRobotScene
from revolve2.modular_robot_simulation._simulate_scenes import simulate_scenes
from revolve2.simulators.mujoco_simulator._local_simulator import LocalSimulator
from revolve2.standards import terrains
from revolve2.standards.simulation_parameters import make_standard_batch_parameters

from project2.individual import Individual
from project2.genotype import Genotype


def load_genotype(genotype_str: str):
    return pickle.loads(codecs.decode(genotype_str.encode(), "base64"))


loaded_json = json.load(
    open(
        "stats/experiment_lowest_fitness_max_distance/run_20250725_113938/stats_generation_99.json"
    )
)


body = load_genotype(
    loaded_json["robot_stats"]["2f91b4d4-693c-11f0-b94d-a221a9d4cbd0"]["genotype"][
        "body"
    ]
)
brain = load_genotype(
    loaded_json["robot_stats"]["2f91b4d4-693c-11f0-b94d-a221a9d4cbd0"]["genotype"][
        "brain"
    ]
)

genotype = Genotype(brain, body)
ind = Individual(genotype, 0)
robot = ind.develop()


local_simulator = LocalSimulator(headless=False, viewer_type="native")

scene = ModularRobotScene(terrain=terrains.flat())
scene.add_robot(robot)

batch_parameters = make_standard_batch_parameters()
batch_parameters.simulation_time = 60  # Here we update our simulation time.

# Simulate the scene.
# A simulator can run multiple sets of scenes sequentially; it can be reused.
# However, in this tutorial we only use it once.
simulate_scenes(
    simulator=local_simulator,
    batch_parameters=batch_parameters,
    scenes=scene,
)
