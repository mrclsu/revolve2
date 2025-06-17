from uuid import UUID
from project2.individual import Individual
import json
import pickle
from pathlib import Path
from datetime import datetime
import codecs


class Statistics:
    def __init__(self, folder_name: str = "stats"):
        self.generation_to_population_size: dict[int, int] = {}
        self.robot_stats: dict[str, dict[str, float]] = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_path = Path(folder_name) / f"run_{timestamp}"
        self.folder_path.mkdir(parents=True, exist_ok=True)

    def add_generation(self, generation: int, population_size: int):
        self.generation_to_population_size[generation] = population_size

    def get_population_size(self, generation: int) -> int:
        return self.generation_to_population_size[generation]

    def track_individuals(
        self, uuid_to_individual: dict[UUID, Individual], generation: int
    ):
        for uuid, individual in uuid_to_individual.items():
            str_uuid = str(uuid)
            if str_uuid not in self.robot_stats:
                self.robot_stats[str_uuid] = {}
            self.robot_stats[str_uuid]["initial_generation"] = (
                individual.initial_generation
            )
            if individual.final_generation != -1:
                self.robot_stats[str_uuid]["final_generation"] = (
                    individual.final_generation
                )
            else:
                self.robot_stats[str_uuid]["final_generation"] = None

            self.robot_stats[str_uuid]["genotype"] = {
                "body": codecs.encode(
                    pickle.dumps(individual.genotype.body), "base64"
                ).decode(),
                "brain": codecs.encode(
                    pickle.dumps(individual.genotype.brain), "base64"
                ).decode(),
            }

            if "fitness" not in self.robot_stats[str_uuid]:
                self.robot_stats[str_uuid]["fitness"] = {}
            self.robot_stats[str_uuid]["fitness"][generation] = individual.fitness

    def flush_to_json(self, postfix: str = ""):
        data = {
            "generation_to_population_size": self.generation_to_population_size,
            "robot_stats": self.robot_stats,
        }
        with open(self.folder_path / f"stats_{postfix}.json", "w+") as f:
            json.dump(data, f, indent=4)
