from copy import deepcopy

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