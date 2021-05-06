from random import sample, uniform
from operator import attrgetter

def random_selection(population):
    """Selects a random individual from the given population

    Args:
        population (Population): The population from which we choose from.

    Returns:
        Individual: Selected individual

    """

    return sample(population.individuals, 1)

def fps(population):
    """Fitness proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.

    Returns:
        Individual: selected individual.
    """
    # Sum total fitnesses
    total_fitness = sum([i.fitness for i in population])
    # Get a 'position' on the wheel
    spin = uniform(0, total_fitness)
    position = 0
    # Find individual in the position of the spin
    for individual in population:
        position += individual.fitness
        if position > spin:
            return individual

def tournament(population, size=5):
    """
    Tournament proportionate selection implementation.

    Args:
        population (Population): The population we want to select from.
        size: The size of the tournament

    Returns:
        Individual: selected individual
    """

    tournament = sample(population.individuals, size)
    if population.optim == 'max':
        return max(tournament, key=attrgetter('fitness'))
    elif population.optim == 'min':
        return min(tournament, key=attrgetter('fitness'))
    else:
        raise Exception("No optimization specified(min or max")

