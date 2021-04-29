from random import shuffle, choice, sample, random
from operator import  attrgetter
import numpy as np
from project.game import main

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
    def __init__(self, size):
        self.individuals = []
        self.size = size
        for _ in range(size):
            self.individuals.append(
                Individual( )
            )
            
    def evolve(self, gens, select, crossover, mutate, co_p, mu_p, elitism):
        for gen in range(gens):
            new_pop = []
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
                raise NotImplementedError

            self.individuals = new_pop
            print(f'Best Individual: {max(self, key=attrgetter("fitness"))}')

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"



if __name__=='__main__':
    pop = Population(
        size=10,
    )

    print(f'Best Individual: {max(pop.self, key=attrgetter("fitness"))}')
