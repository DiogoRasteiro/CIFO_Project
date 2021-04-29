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