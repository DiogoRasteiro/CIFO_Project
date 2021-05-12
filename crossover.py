from random import random, randint, sample
from copy import deepcopy

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

def standard_co(p1, p2):
    """Implementation of a standard crossover.
    
    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.
    
    Returns:
        Individuals: Two offspring, resulting from the crossover."""

    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)

    co_point=randint(0,len(p1)-1)

    offspring1 = p1[:co_point] + p2[co_point:]
    offspring2 = p2[:co_point] + p1[co_point:]


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

def pmx_co(p1, p2):
    """Implementation of Partially Matched crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """
    # Sample 2 random co points
    co_points = sample(range(len(p1)), 2)
    co_points.sort()

    def PMX(x, y):
        # Create placeholder for offspring
        o = [None] * len(x)

        # Copy co segment into offspring
        o[co_points[0]:co_points[1]] = x[co_points[0]:co_points[1]]

        # Find set of values not in offspring from co segment in P2
        z = set(y[co_points[0]:co_points[1]]) - set(x[co_points[0]:co_points[1]])

        # Map values in set to corresponding position in offspring
        for i in z:
            temp = i
            index = y.index(x[y.index(temp)])
            while o[index] != None:
                temp = index
                index = y.index(x[temp])
            o[index] = i
        
        while None in o:
            index = o.index(None)
            o[index] = y[index]
        return o

    o1, o2 = (
        PMX(p1, p2),
        PMX(p2, p1)
    )

    return o1, o2

def cycle_co(p1, p2):

    """Implementation of geometric crossover.

    Args:
        p1 (Individual): First parent for crossover.
        p2 (Individual): Second parent for crossover.

    Returns:
        Individuals: Two offspring, resulting from the crossover.
    """

    # Offspring placeholders - None values make it easy to debug for errors
    offspring1 = [None] * len(p1)
    offspring2 = [None] * len(p1)
    # While there are still None values in offspring, get the first index of
    # None and start a "cycle" according to the cycle crossover method
    while None in offspring1:
        index = offspring1.index(None)
        # alternate parents between cycles beginning on second cycle
        if index != 0:
            p1, p2 = p2, p1
        val1 = p1[index]
        val2 = p2[index]

        while val1 != val2:
            offspring1[index] = p1[index]
            offspring2[index] = p2[index]
            val2 = p2[index]
            index = p1.index(val2)
        # In case last values share the same index, fill them in each offspring
        offspring1[index] = p1[index]
        offspring2[index] = p2[index]

    return offspring1, offspring2
