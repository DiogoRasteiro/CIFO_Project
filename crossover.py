from random import random
from copy import deepcopy

def template_co(p1, p2):
    """[summary]

    Args:
        p1 ([type]): [description]
        p2 ([type]): [description]

    Returns:
        [type]: [description]
    """

    return offspring1, offspring2

def test_crossover(p1, p2):
    """A crossover for testing the algorithm that just copies the parents

    Args:
        p1 (Individual): The first parent
        p2 (Individual): The second parent

    Returns:
        Offspring: Tuple of two offspring that are exact copies of the parents
    """
    offspring1 = deepcopy(p1)
    offspring2 = deepcopy(p2)

    return offspring1, offspring2

def geometric_co(p1, p2):
    """Implementation of geometric crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    a = random()
    b = random()

    for i in range(len(p1)):
        offspring1[i] = a*p1[i] + (1-a)*p2[i]
        offspring2[i] = b*p1[i] + (1-b)*p2[i]


    return offspring1, offspring2

