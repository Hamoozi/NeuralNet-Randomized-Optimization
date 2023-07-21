# -*- coding: utf-8 -*-
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import matplotlib.pyplot as plt
import time



fitness = mlrose.Queens()
problem = mlrose.DiscreteOpt(length = 8, fitness_fn = fitness, maximize=True, max_val=8)
schedule = mlrose.ExpDecay()

init_state = np.array([0, 1, 2, 3, 4, 5, 6, 7])
fitness_table = []

st = time.time()
simstate, simfitness = mlrose.simulated_annealing(problem, schedule = schedule, max_attempts = 100, 
                                                      max_iters = 10000, init_state = init_state,
                                                      random_state = 3)
end = time.time()
st1 = time.time()
genstate, genfitness = mlrose.genetic_alg(problem, mutation_prob=0.2, max_attempts=10, max_iters=1000, random_state=3)
end1 = time.time()





print("Genetic Fitness: {}, Time: {}".format(genfitness, end1-st1))
print("Simulated Annealing Fitness: {}, Time: {}".format(simfitness, end-st))
