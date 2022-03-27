from main_alns import draw_animated_output
from drone_vrp import *
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt
import time

def PrintCustomers(customers):
    s = "("
    for custom in customers:
        s+=str(custom.id) + ', '
    s= s[0:-2]+")"
    return s;


def rankRoutes(population, dvrp):
    fitnessResults = {}
    for i in range(0,len(population)):
        #print('population id:')
        #for custom in population[i]:
        #    print(custom.id)
        dvrp1 = copy.deepcopy(dvrp)
        dvrp1.customers = population[i]
        dvrp1.split_route();
        fitnessResults[i] = 1/dvrp1.objective()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

'''
Description: Main method of GA
Reference: 
https://levelup.gitconnected.com/how-to-implement-a-traveling-salesman-problem-genetic-algorithm-in-python-ea32c7bef20f
https://github.com/rocreguant/personal_blog/blob/main/Genetic_Algorithm_Python_Example/Traveling_Salesman_Problem.ipynb

'''

## Create our initial population
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    
    return route

#Create first "population" (list of routes)
def initialPopulation(popSize, cityList):
    population = []

    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

#Create mating pool
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

#Create a crossover function for two parents to create one child
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#Create a crossover function for two parents to create one child
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

#Create function to run crossover over full mating pool
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    for i in range(0,len(children)):
        print("child",i,PrintCustomers(children[i]))
    return children

#Create function to mutate a single route
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#Create function to run mutation over entire population
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    for i in range(0,len(mutatedPop)):
        print("mutate child",i,PrintCustomers(mutatedPop[i]))
    return mutatedPop

#Put all steps together to create the next generation
def nextGeneration(currentGen, eliteSize, mutationRate, otherParamters):
    popRanked = rankRoutes(currentGen, otherParamters)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration,popRanked

#Final step: create the genetic algorithm
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations, dvrp):
    print(population)
    pop = initialPopulation(popSize, population)
    for i in range(0,len(pop)):
        print("Initial pop",i,PrintCustomers(pop[i]))
    popRanked = rankRoutes(pop, dvrp);
    print("Initial Service Time: " + str(1 / popRanked[0][1]))
    progress = []
    progress.append(1 / popRanked[0][1])
    for i in range(0, generations):
        print('start generations:', i+1)
        pop,popRanked = nextGeneration(pop, eliteSize, mutationRate, dvrp)
        
        progress.append(1 / popRanked[0][1])
    
    print("Final Service Time: " + str(1 / popRanked[0][1]))
    plt.plot(progress)
    plt.ylabel('Service Time')
    plt.xlabel('Generation')
    plt.show()
    bestRouteIndex = popRanked[0][0]
    bestRoute = pop[bestRouteIndex]
    dvrp1 = copy.deepcopy(dvrp)
    dvrp1.customers = bestRoute
    dvrp1.split_route();
    
    return dvrp1



if __name__ == '__main__':
    # instance file and random seed
    config_file = "config.ini"
    data_type = "data-medium"
    
    # # load data and random seed
    parsed = Parser(config_file, data_type)
    
    dvrp = DVRP(parsed.warehouses, parsed.customers, parsed.trucks, parsed.drones, parsed.map_size)


    start = time.time()

    
    bestRoute = geneticAlgorithm(population=dvrp.customers, popSize=20, eliteSize=2, mutationRate=0.01, generations=10,dvrp=dvrp)
    
    end = time.time()
    print('running time: ',end - start)
    


    



