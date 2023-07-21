import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#https://stackoverflow.com/questions/61867945/python-import-error-cannot-import-name-six-from-sklearn-externals
#Fixed six import issue using stackoverflow from above
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time


data = pd.read_csv("file")
X = data.iloc[:,2:-1]
Y = data.iloc[:,-1:]
Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y,test_size=.2, random_state=0)
scaler = MinMaxScaler()
print()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)
randhillacc_table = []
simacc_table = []
genacc_table = []

randhillacc_test = []
simacc_test = []
genacc_test = []


#Placing in loop allows for a accurate plot graph    
for i in range(1,1000,200):
    

    model = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                     algorithm = 'random_hill_climb', max_iters = i, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     early_stopping = True, clip_max = 2, max_attempts = 100, \
                                     random_state = 3)          


    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtrain)
    trainval_x = accuracy_score(ytrain, pred)
    randhillacc_table.append(trainval_x)

    testpred = model.predict(Xtest)
    testval_x = accuracy_score(ytest, testpred)
    randhillacc_test.append(testval_x)     

    modely = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                     algorithm = 'simulated_annealing', max_iters = i, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     early_stopping = True, clip_max = 2, max_attempts = 100, \
                                     random_state = 3)
   
    
    modely.fit(Xtrain, ytrain)
    pred = modely.predict(Xtrain)
    trainval_y = accuracy_score(ytrain, pred)
    simacc_table.append(trainval_y)   
    
    testpred = modely.predict(Xtest)
    testval_y = accuracy_score(ytest, testpred)
    simacc_test.append(testval_y)
    
    
    modelz = mlrose.NeuralNetwork(hidden_nodes = [2], activation = 'relu', \
                                     algorithm = 'genetic_alg', max_iters = i, \
                                     bias = True, is_classifier = True, learning_rate = 0.1, \
                                     early_stopping = True, clip_max = 2, max_attempts = 100, \
                                     random_state = 3)
   
    modelz.fit(Xtrain, ytrain)
    pred = modelz.predict(Xtrain)
    trainval_z = accuracy_score(ytrain, pred)
    genacc_table.append(trainval_z)
    
    testpred = modelz.predict(Xtest)
    testval_z = accuracy_score(ytest, testpred)
    genacc_test.append(testval_z)

    
    

plt.plot(np.arange(1, 1000,200), np.array(randhillacc_table), label='Random Hill Climb')
plt.plot(np.arange(1, 1000,200), np.array(simacc_table), label='Simulated Annealing')
plt.plot(np.arange(1, 1000,200), np.array(genacc_table), label='Genetic Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Training Acccuracy')
plt.title('Training Rates')
plt.legend()
plt.show()
print(np.array(model))
print(accuracy_score(ytrain, pred))

print("RHC {}, SIM {}, GEN {}.".format(trainval_x, trainval_y, trainval_z))

plt.figure()
plt.plot(np.arange(1, 1000,200), np.array(randhillacc_test), label='Random Hill Climb')
plt.plot(np.arange(1, 1000,200), np.array(simacc_test), label='Simulated Annealing')
plt.plot(np.arange(1, 1000,200), np.array(genacc_test), label='Genetic Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Test Accuracy')
plt.title('Testing Rates')
plt.legend()
plt.show()



#From mlrose
def random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                      init_state=None, curve=False, random_state=None):
    
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -1*np.inf
    best_state = None

    if curve:
        fitness_curve = []

    for _ in range(restarts + 1):
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1

            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

            if curve:
                fitness_curve.append(problem.get_fitness())

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness
#From mlrose
def simulated_annealing(problem, schedule=mlrose.GeomDecay(), max_attempts=10,
                        max_iters=np.inf, init_state=None, curve=False,
                        random_state=None):
    
    
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    if curve:
        fitness_curve = []

    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)

            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter
            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness
#From mlrose
def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf, curve=False, random_state=None):

    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    if curve:
        fitness_curve = []

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

        if curve:
            fitness_curve.append(problem.get_fitness())

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness



