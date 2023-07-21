
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import time
import matplotlib.pyplot as plt

locationcoord = [(1,1),(2,3),(4,2),(3,1),(5,5),(2,2),(3,4),(1,5)]

fitnesscoords = mlrose.TravellingSales(coords = locationcoord)

problem = mlrose.TSPOpt(length = 8, fitness_fn = fitnesscoords, maximize = True)


st = time.time()
simstate, simfitness = mlrose.simulated_annealing(problem, max_attempts = 10, max_iters = 1000, random_state = 3)
end = time.time()
st1 = time.time()
genstate, genfitness = mlrose.genetic_alg(problem, mutation_prob=0.2, max_attempts=10, max_iters=1000, random_state=3)

end1 = time.time()



print("Genetic Fitness: {}, Time: {}".format(genfitness, end1-st1))
print("Simulated Annealing Fitness: {}, Time: {}".format(simfitness, end-st))