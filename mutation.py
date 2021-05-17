from copy import deepcopy
from random import random, uniform, sample
import numpy as np
from operator import add

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

    r_weights = np.random.uniform(low=-.3, high=.3, size=(len(individual),))

    r_weights = [0 if random() < 0.9 else weight for weight in r_weights] 
    
    # To avoid changing the weights of the original, we perform a deepcopy
    fetus = deepcopy(individual)

    fetus = list(map(add, fetus, r_weights))

    return fetus


def swap_mutation(individual):

    """Implementation of swap mutation.

    Args:
        individual (Individual): The original individual to be mutated

    Returns:
        Individual: A new individual resulting from geometric mutation of the original
    """

    # Get two mutation points
    mut_points = sample(range(len(individual)), 2)
    # Rename to shorten variable name
    i = individual

    i[mut_points[0]], i[mut_points[1]] = i[mut_points[1]], i[mut_points[0]]

    return i


def inversion_mutation(individual):
    """Implementation of swap mutation.

    Args:
        individual (Individual): The original individual to be mutated

    Returns:
        Individual: A new individual resulting from geometric mutation of the original
    """

    i = individual
    # Position of the start and end of substring
    mut_points = sample(range(len(i)), 2)
    # This method assumes that the second point is after (on the right of) the first one
    # Sort the list
    mut_points.sort()
    # Invert for the mutation
    i[mut_points[0]:mut_points[1]] = i[mut_points[0]:mut_points[1]][::-1]

    return i

