from random import shuffle, choice, sample, random
from operator import  attrgetter
import numpy as np
from project.game import main
from copy import deepcopy

from selection import random_selection
from crossover import test_crossover
from mutation import test_mutation

def create_weights():
	first_layer = np.random.rand(16,16)
	second_layer = np.random.rand(16,)
	third_layer = np.random.rand(16,64)
	fourth_layer = np.random.rand(64,)
	fifth_layer = np.random.rand(64,4)
	sixth_layer = np.random.rand(4,)

	return np.array((first_layer, second_layer, third_layer, fourth_layer, fifth_layer, sixth_layer))

class Individual:
    def __init__(
        self,
        representation=None,
    ):
        if representation == None:
            self.representation = create_weights()
        else:
            self.representation = representation
        self.fitness = self.evaluate()

    def evaluate(self):
        return main(self.representation)

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim):
        self.individuals = []
        self.size = size
        self.optim = optim
        for _ in range(size):
            self.individuals.append(
                Individual( 
                )
            )
            
    def evolve(self, gens, select, elitism, mutate, crossover, mu_p, co_p):
    #def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism):
        for gen in range(gens):
            new_pop = []

            if elitism == True:
                if self.optim == 'max':
                    elite = deepcopy(max(self.individuals, key=attrgetter('fitness')))
                elif self.optim == 'min':
                    elite = deepcopy(min(self.individuals, key=attrgetter('fitness')))

            while len(new_pop) < self.size:

                parent1, parent2 = select(self), select(self)
                # Crossover
                if random() < co_p:
                    offspring1, offspring2 = crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                # Mutation
                if random() < mu_p:
                    offspring1 = mutate(offspring1)
                if random() < mu_p:
                    offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2))

            if elitism == True:
                if self.optim == 'max':
                    least = min(new_pop.individuals, key=attrgetter('fitness'))
                elif self.optim == 'min':
                    least = max(new_pop, key=attrgetter('fitness'))
                new_pop.append(elite)
                new_pop.pop((new_pop.pop.index(least)))

            self.individuals = new_pop

            if self.optim == 'max':
                print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')
            elif self.optim == 'min':
                print(f'Best Individual: {min(self, key=attrgetter("fitness"))}')


    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"



if __name__=='__main__':
    pop = Population(
        size=10,
        optim = 'max'
    )


    pop.evolve(
        gens=3, 
        select= random_selection,
        crossover= test_crossover,
        mutate=test_mutation,
        co_p=0.7,
        mu_p=0.2,
        elitism=True,
    )

    print(f'Best Individual: {max(pop.individuals, key=attrgetter("fitness"))}')