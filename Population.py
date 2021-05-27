from random import shuffle, choice, sample, random
from operator import  attrgetter
import numpy as np
import pandas as pd
from project.game import main
from copy import deepcopy
from utils import *
import os
import math

from selection import *
from crossover import  *
from mutation import *

def create_weights():
    """Function to create a random set of weights for our neural network, AKA the genotype of our individuals.

    Since the focus of this work is to optimize using a GA, the architecture of the Neural Network is preset. Thus we decided to directly
    generate it. 
    The NN library is made with Keras, who deals with weights as an np.array with 2 sub-arrays for each layer(one for bias and the other
    for weights). Thus we generate each sub-array individually and then join them together

    Returns:
        weights: A numpy array with 6 sub-arrays, each with a different shape and size, of random values between -1 and 1.
    """
    first_layer = np.random.rand(16,16) * np.random.choice([-1,1])
    second_layer = np.random.rand(16,) * np.random.choice([-1,1])
    third_layer = np.random.rand(16,64) * np.random.choice([-1,1])
    fourth_layer = np.random.rand(64,)* np.random.choice([-1,1])
    fifth_layer = np.random.rand(64,4)* np.random.choice([-1,1])
    sixth_layer = np.random.rand(4,)* np.random.choice([-1,1])
    return np.array((first_layer, second_layer, third_layer, fourth_layer, fifth_layer, sixth_layer))

class Individual:
    def __init__(self, fitness_type, representation=None, reevaluate=False):
        """Initialization method for an individual

        Args:
            - fitness_type: Type of scoring to be used to evaluate an individual. This value will be passed to the fitness function to 
            determine which of the values returned from the game will be considered the fitness.
            - representation: Genotype to be set as an individual. This is expected to be None at the start of the population, where it will
            be randomly generated with the create_weights() function. The individuals of the next generations are expected to be passed
            a genotype resulting from GA operations of the parents.
            - reevaluate: When this parameter is set to true, the individual will be evaluated 3 times and the final fitness is the average
            of the attempts. This results in the evaluation to be more accurate to an individual's skill, but the running time will drastically
            increase.
        """
        self.fitness_type = fitness_type
        self.reevaluate = reevaluate

        if representation is None:
            # If we are creating an individual from scratch, create a random genotype
            # This genotype is flattened to make it easier to perform GA operations
            self.representation = flatten(create_weights())
        else:
            self.representation = representation

        # Always evaluate an individual on creation and store its fitness as an attribute.
        self.fitness = self.evaluate()

    def evaluate(self):
        """ Function that evaluates a certain individual.

        Returns:
            - fitness: A value that represents how well an individual has done. This fitness can be multiple types, according to the
            individual's fitness_type attribute:
                - score: Standard 2048 game score. Essentially adds the value of the pieces that are merged together.
                - max_Tile: Max tile that has been merged.
                - num_moves: number of moves the individual achieved before losing/getting stuck
                - combined: our custom formula that, with the score as basis, benefits individuals who reached high tiles and penalizes
                those who used too many moves.
        """

        if self.reevaluate:
            # If this parameter is true, we will evaluate the individual 3 times and return the averages

            # First some empty arrays to store the data
            max_tiles=np.empty(3)
            score=np.empty(3)
            moves=np.empty(3)

            for i in range(3):
                # Get the data from the game
                game_score = main(unflatten(self.representation), display_graphics=True)
                # And store it in the respective arrays
                max_tiles[i]=game_score['max_tile']
                score[i]=game_score['score']
                moves[i]=game_score['num_moves']

            # Since tiles are always multiples of 2, we will use the median instead to ensure it is a valid value
            max_tiles=np.sort(max_tiles)
            game_score['max_tile']=max_tiles[1] # Since we are only considering three iterations, we can directly pick the middle value

            # Then the means
            game_score['score']=np.mean(score)
            game_score['num_moves']=np.mean(moves)
            game_score['combined']=math.log2(game_score['max_tile'])*(game_score['score']**2)*(game_score['score']/game_score['num_moves'])
        else:
            game_score = main(unflatten(self.representation), display_graphics=True)
            game_score['combined']=math.log2(game_score['max_tile'])*(game_score['score']**2)*(game_score['score']/(game_score['num_moves']+1))
        
        # Finally, we save all the scores and take the selected one as the fitness
        self.game_score = game_score
        return self.game_score[self.fitness_type]

    def __len__(self):
        return len(self.representation)

    def __getitem__(self, position):
        return self.representation[position]

    def __setitem__(self, position, value):
        self.representation[position] = value

    def __repr__(self):
        return f"Individual(size={len(self.representation)}); Fitness: {self.fitness}"


class Population:
    def __init__(self, size, optim, fitness_type, reevaluate=False):
        """Initialization method for a Population.

        Args:
            - size: Number of individuals per generation
            - optim: A string value that is either "max" or "min", that respectively represents whether the GA will try to maximize
            population fitness or minimize it.
            - fitness_type: Which type of fitness to consider for the Individuals. See the Individual class for more information.
            - reevaluate: Whether to evaluate an individual multiple times or not. See the Individual class for more information.
        """

        self.individuals = []
        self.size = size
        self.optim = optim
        self.fitness_type = fitness_type
        self.reevaluate = reevaluate

        # On initialization, create a population of random individuals
        for _ in range(size):
            self.individuals.append(
                Individual(
                    self.fitness_type,
                    reevaluate = self.reevaluate
                )
            )
            
    def evolve(self, gens, select, elitism, mutate, crossover, mu_p, co_p, decay_rate, export_data = True, export_champion=False):
        """ Function to use GA operators on a Population to improve fitness

        Args:
            - gens: Number of generations to evolve for.
            - select: Selection function to use. See the "selection.py" file for details.
            - mutate: Mutation function to use. See the "mutation.py" file for details.
            - crossover: Crossover function to use. See the "crossover.py" file for details.
            - elitism: Boolean on whether a generation's best individual should be directly copied to the next one. Due to the random nature
            - of the game, this individual is also evaluated again to ensure they did not simply get a "lucky" seed.
            - mu_p: Probability of mutation occuring. Should be a boolean between 0 and 1.
            - co_p: Probability of crossover occuring. Should be a boolean between 0 and 1.
            - decay_rate: Parameter exclusive to the geometric mutation with decay. Controls how the learning rate decreases over time.
            - export_data: Boolean to decide whether we should return the data on the GA. This data is in the format of a csv file that
            contains, for each generation, all of the scores for the best individual.
            - export_champion: Boolean to decide whether to return the final best Individual.
        
        Returns:
            - Either:
                - Nothing, if neither export_data or export_champion is true.
                - A list of lists, where the inner lists contain the various scores for each generation's best individuals, if only 
                export_data = True.
                - An Individual if only export_champion is true.
                - Both the list of lists and an Individual if both booleans are true.
        """

        # List to store the generations
        csv_row=[]
        # Learning rate to be used with geometric mutation with decay.
        lr=1

        # Iterate over all generations
        for gen in range(gens):
            new_pop = []

            if elitism == True: # Save the best Individual if elitism
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
                # Since geometric_mu_decay needs to have its learning rate controlled, it must be called a bit differently
                if mutate==geometric_mu_decay:  
                    if random() < mu_p:
                        offspring1 = mutate(offspring1, lr)
                    if random() < mu_p:
                        offspring2 = mutate(offspring2, lr)
                else:
                    if random() < mu_p:
                        offspring1 = mutate(offspring1)
                    if random() < mu_p:
                        offspring2 = mutate(offspring2)

                new_pop.append(Individual(representation=offspring1, fitness_type=self.fitness_type))
                if len(new_pop) < self.size:
                    new_pop.append(Individual(representation=offspring2, fitness_type=self.fitness_type))

            # In case of elistim, we must remove the worst Individual to make room for the elite
            if elitism == True:
                if self.optim == 'max':
                    least = min(new_pop, key=attrgetter('fitness'))
                elif self.optim == 'min':
                    least = max(new_pop, key=attrgetter('fitness'))

                # Then we insert the elite that is carried over from the previous generation
                # Notice they are recalculated to lessen the effect of having a "lucky" seed
                new_pop.append(Individual(representation=elite.representation, fitness_type=self.fitness_type))
                new_pop.pop((new_pop.index(least)))

            self.individuals = new_pop

            print(f'Best member of gen {gen+1}')

            # Here we find the best indvidual of this generation
            # If export_data is true, we also add their info
            if self.optim == 'max':
                champion = max(self, key=attrgetter("fitness"))
                if export_data:
                    csv_row.append(list(champion.game_score.values()))
                print(champion.game_score)

            elif self.optim == 'min':
                champion = min(self, key=attrgetter("fitness"))
                if export_data:
                    csv_row.append(list(champion.game_score.values()))
                print(champion.game_score)

            # Adjusting the learning rate, only applicable to the geometric mutation with decay
            lr=lr*decay_rate
        
        # Depending on the parameters, different things are returned
        if export_champion and not export_data:
            return champion
        elif export_data and not export_champion:
            return csv_row
        elif export_champion and export_data:
            return champion, csv_row
        else:
            return None

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, position):
        return self.individuals[position]

    def __repr__(self):
        return f"Population(size={len(self.individuals)}, individual_size={len(self.individuals[0])})"
