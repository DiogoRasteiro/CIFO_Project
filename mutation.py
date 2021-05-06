from copy import deepcopy
from random import uniform
import numpy as np
from operator import add

def template_mutation(individual):
    """[summary]

    Args:
        individual ([type]): [description]

    Returns:
        [type]: [description]
    """
    return individual

def test_mutation(individual):
    """Mutation that simply copies the individual, used for testing purposes

    Args:
        individual (Individual): The original individual to be mutated.

    Returns:
        i(Individual): A copy of the original individual
    """

    i = deepcopy(individual)

    return i

def geometric_mutation(individual):
    """Implementation of geometric mutation. This implementation generates an array of len(individual)
    random values and performs an elementwise addition.

    Args:
        individual (Individual): The original individual to be mutated

    Returns:
        Individual: A new individual resulting from geometric mutation of the original
    """

    r_weights = np.random.uniform(low=-0.5, high=0.5, size=(len(individual),))
    
    # To avoid changing the weights of the original, we perform a deepcopy
    fetus = deepcopy(individual)

    fetus = list(map(add, fetus, r_weights))

    return fetus
