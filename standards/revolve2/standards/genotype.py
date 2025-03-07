"""Genotype class."""

from __future__ import annotations

from dataclasses import dataclass

import multineat
import numpy as np
import numpy.typing as npt


from revolve2.standards import config
from revolve2.modular_robot import ModularRobot
from revolve2.standards.genotypes.cppnwin._multineat_genotype_pickle_wrapper import MultineatGenotypePickleWrapper
from revolve2.standards.genotypes.cppnwin.modular_robot import BrainGenotypeCpg
from revolve2.standards.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeV2



@dataclass
class Genotype(BodyGenotypeV2, BrainGenotypeCpg):
    """A genotype for a body and brain using CPPN."""
    parameters: npt.NDArray[np.float_]

    @classmethod
    def random(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Create a random genotype.

        :param innov_db_body: Multineat innovation database for the body. See Multineat library.
        :param innov_db_brain: Multineat innovation database for the brain. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        body = cls.random_body(innov_db_body, rng)
        brain = cls.random_brain(innov_db_brain, rng)

        return Genotype(body=body.body, brain=brain.brain)

    def develop(self, visualize: bool = False) -> ModularRobot:
        """
        Develop the genotype into a modular robot.

        :param visualize: Wether to plot the mapping from genotype to phenotype.
        :returns: The created robot.
        """
        body = self.develop_body(visualize=visualize)
        brain = self.develop_brain(body=body)
        return ModularRobot(body=body, brain=brain)
    
    def mutate(
        self,
        rng: np.random.Generator,
    ) -> Genotype:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        return Genotype(
            rng.normal(scale=config.MUTATE_STD, size=config.NUM_PARAMETERS)
            + self.parameters
        )
    
    @classmethod
    def crossover(
    cls,
    parent1: Genotype,
    parent2: Genotype,
    rng: np.random.Generator,
) -> Genotype:
        """
        Perform uniform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
         # Extract body and brain genomes from the parents
        parent1_body_genome = parent1.body.genotype
        parent2_body_genome = parent2.body.genotype
        parent1_brain_genome = parent1.brain.genotype
        parent2_brain_genome = parent2.brain.genotype

        # Prepare the NEAT Parameters object (can be specific to your setup)
        parameters = multineat.Parameters()  # This could be your default parameters

        # Mate the body genomes (average and interspecies flags can be adjusted as needed)
        new_body_genome = parent1_body_genome.Mate(
            parent2_body_genome,  # Second parent body genome
            True,                 # Mate average (adjustable based on your requirements)
            False,                # Inter-species (adjustable based on your requirements)
            rng,                  # Random number generator
            parameters            # NEAT Parameters
        )

        # Mate the brain genomes similarly
        new_brain_genome = parent1_brain_genome.Mate(
            parent2_brain_genome,  # Second parent brain genome
            True,                  # Mate average (adjustable)
            False,                 # Inter-species (adjustable)
            rng,                   # Random number generator
            parameters             # NEAT Parameters
        )

        # Wrap the new genomes into MultineatGenotypePickleWrapper
        new_body_wrapper = MultineatGenotypePickleWrapper(new_body_genome)
        new_brain_wrapper = MultineatGenotypePickleWrapper(new_brain_genome)

        # Return the new genotype combining the body and brain
        return Genotype(
        body=new_body_wrapper, 
        brain=new_brain_wrapper,
        parameters=parameters  # Pass the parameters argument here
    )

