from uuid import UUID
from individual import Individual
import json


class Statistics:
    def __init__(self, folder: str):
        self.generation_to_population_size: dict[int, int] = {}
        self.robot_stats: dict[str, dict[str, float]] = {}
        self.folder = folder

    def add_generation(self, generation: int, population_size: int):
        self.generation_to_population_size[generation] = population_size

    def get_population_size(self, generation: int) -> int:
        return self.generation_to_population_size[generation]

    def track_individuals(self, uuid_to_individual: dict[UUID, Individual]):
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

    def flush_to_json(self, postfix: str):
        data = {
            "generation_to_population_size": self.generation_to_population_size,
            "robot_stats": self.robot_stats,
        }
        with open(f"{self.folder}/stats_{postfix}.json", "w+") as f:
            json.dump(data, f, indent=4)
