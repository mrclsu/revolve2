from typing import Any

from revolve2.experimentation.evolution.abstract_elements import (
    Evolution,
)


from .parent_selector import ParentSelector
from .survivor_selector import SurvivorSelector
from .evaluator import Evaluator
from .crossover_reproducer import CrossoverReproducer


TPopulation = (
    Any  # An alias for Any signifying that a population can vary depending on use-case.
)


class ModularRobotEvolution(Evolution):
    """An object to encapsulate the general functionality of an evolutionary process for modular robots."""

    _parent_selection: ParentSelector
    _survivor_selection: SurvivorSelector
    _evaluator: Evaluator
    _reproducer: CrossoverReproducer

    def __init__(
        self,
        parent_selection: ParentSelector,
        survivor_selection: SurvivorSelector,
        evaluator: Evaluator,
        reproducer: CrossoverReproducer,
    ) -> None:
        """
        Initialize the ModularRobotEvolution object to make robots evolve.

        :param parent_selection: Selector object for the parents for reproduction.
        :param survivor_selection: Selector object for the survivor selection.
        :param evaluator: Evaluator object for evaluation.
        :param reproducer: The reproducer object.
        """
        self._parent_selection = parent_selection
        self._survivor_selection = survivor_selection
        self._evaluator = evaluator
        self._reproducer = reproducer

    def step(self, population: TPopulation, **kwargs: Any) -> TPopulation:
        """
        Step the current evolution by one iteration.

        This implementation follows the following schedule:

            [Parent Selection] ---------> [Reproduction]

                   ^                             |
                   |                             |
                   |                             ⌄

            [Survivor Selection] <----- [Evaluation of Children]

        The schedule can be easily adapted and reorganized for your needs.

        :param population: The current population.
        :param kwargs: Additional keyword arguments to use in the step.
        :return: The population resulting from the step
        """
        parents, parent_kwargs = self._parent_selection.select(population, **kwargs)
        merged_kwargs = {**parent_kwargs, **kwargs}
        children, parent_uuids = self._reproducer.reproduce(parents, **merged_kwargs)
        child_task_performance, child_all_fitness_metrics = self._evaluator.evaluate(
            children
        )
        survivors, *_ = self._survivor_selection.select(
            population,
            **kwargs,
            children=children,
            child_task_performance=child_task_performance,
            child_all_fitness_metrics=child_all_fitness_metrics,
            parent_uuids=parent_uuids,
        )
        return survivors
