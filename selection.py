from random import sample

def random_selection(population):
    """Selects a random individual from the given population

    Args:
        population (Population): The population from which we choose from.

    Returns:
        Individual: Selected individual

    """

    return sample(population.individuals, 1)