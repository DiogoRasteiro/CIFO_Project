# The basis for this code is the 'charles' library made by professor Davide Montali during the practical lessons
# We made some alterations to implement some logic important to our particular use case. 

import pandas as pd
from utils import *
import os

from selection import *
from crossover import  *
from mutation import *
from Population import Population


if __name__=='__main__':

    # Get current directory to save data to
    wd=os.getcwd()
    # When saving to csv, insert the data 
    path=os.path.join(wd, "fps_standard_xo_geometric_score.csv")
    # Create an empty list which will be used to save the GAs data
    results=[]


    # Each time, create a brand new population with the same parameters
    pop = Population(
        size=10,
        optim = 'max',
        fitness_type = 'score',
        reevaluate=False
    )

    # And evolve it
    results=pop.evolve(
        gens=3, 
        select= fps, 
        crossover= standard_co,
        mutate=geometric_mutation,
        co_p=.9,
        mu_p=0.05,
        decay_rate=0.999,
        elitism=True,
        export_champion=True,
        export_data=True
    )


    # If we chose to export data in the pop parameters, to_csv will have data in it, which we will save 
    if len(results) > 0:
        if len(results) == 2:
            df_to_csv=pd.DataFrame(data=results[1])
            df_to_csv.to_csv(path)
        else:
            df_to_csv=pd.DataFrame(data=results)
            df_to_csv.to_csv(path)
