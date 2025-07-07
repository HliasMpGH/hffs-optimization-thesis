import random
import numpy as np

class PopulationGeneration:

    def __init__(self, population_size, valid_genes):
        # validate given parameters
        self.__validate_parameters(population_size, valid_genes)

        self.population_size = population_size
        self.valid_genes = valid_genes

    def __validate_parameters(self, population_size, valid_genes):
        if population_size <= 0 or type(population_size) != int:
            raise ValueError(
                f"Expected positive integer for `population_size` but got {population_size}"
            )

    '''
    Yield the population randomly.
    The resulting population will contain random
    permutations of the orders
    '''
    def random(self):
        return [np.random.permutation(self.valid_genes) for _ in range(self.population_size)]