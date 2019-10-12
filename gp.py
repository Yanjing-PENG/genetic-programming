# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import os

class GP(object):
    def __init__(self, file_name, size):
        self.data = np.loadtxt(file_name, dtype=np.float, delimiter=',')    # gp training set
        self.size = size
        self.iteration = 0

    # initialize population
    def init_population(self):
        w = np.random.uniform(-1, 1, (self.size, self.data.shape[1] - 1))
        theta = np.random.uniform(-1, 1, self.size)
        self.population = np.insert(w, w.shape[1], values=theta, axis=1)

    # calculate the value of TLU
    def calculate(self, w, x, theta):
        if np.dot(w, x.T) >= theta:
            return 1
        else:
            return 0

    # calculate the fitness value of an individual
    def fitness(self, individual):
        fit = 0
        for i in range(self.data.shape[0]):
            if self.calculate(individual[:-1], self.data[i][:-1], individual[-1]) != self.data[i][-1]:
                fit = fit + 1
        return fit

    # choose the fittest individual among random 10 individuals
    def tournament(self):
        competitors = np.random.randint(0, self.size, 10)
        fittest = self.population[competitors[0]]
        fittest_value = self.fitness(fittest)
        for i in competitors[1:]:
            fitness_value = self.fitness(self.population[i])
            if fitness_value < fittest_value:
                fittest = self.population[i]
                fittest_value = fitness_value
        return fittest

    # get offspring by copy operator
    def copy(self):
        offspring_copy = np.zeros((int(self.size * 0.1), self.data.shape[1]))
        for i in range(offspring_copy.shape[0]):
            offspring_copy[i] = self.tournament()
        return offspring_copy

    # get offspring by crossover operator
    def crossover(self):
        offspring_crossover = np.zeros((int(self.size * 0.89), self.data.shape[1]))
        for i in range(offspring_crossover.shape[0]):
            mother = self.tournament()
            father = self.tournament()
            gene = np.random.randint(0, self.population.shape[1])
            mother[gene] = father[gene]
            offspring_crossover[i] = mother
        return offspring_crossover

    # get offspring by mutation operator
    def mutation(self):
        offspring_mutation = np.zeros((int(self.size * 0.01), self.data.shape[1]))
        for i in range(offspring_mutation.shape[0]):
            mutator = self.tournament()
            gene = np.random.randint(0, self.population.shape[1])
            if gene < self.population.shape[1] - 1:
                mutator[gene] = np.random.uniform(-1, 1, 1)
            else:
                mutator[gene] = np.random.uniform(-9, 9, 1)
            offspring_mutation[i] = mutator
        return offspring_mutation

    # when an individual's fitness value equals to 0, stop iteration
    def stop_condition(self, offspring):
        for individual in offspring:
            if self.fitness(individual) == 0:
                return True
        return False

    def train(self):
        while True:
            offspring_copy = self.copy()
            offspring_crossover = self.crossover()
            offspring_mutation = self.mutation()
            offspring = np.vstack((offspring_copy, offspring_crossover, offspring_mutation))

            # determine whether to stop iteration
            if self.stop_condition(offspring):
                break
            else:
                self.population = offspring
            self.iteration += 1

        return offspring


if __name__ == '__main__':

    # train the TLU by genetic programming
    gp = GP(os.getcwd() + '/data/' + 'gp-training-set.csv', 5000)
    gp.init_population()
    result = gp.train()

    # print out and save results
    column_names = ['w1', 'w2', 'w3', 'w4', 'w5',
                    'w6', 'w7', 'w8', 'w9', 'theta']
    final_result = pd.DataFrame(result, columns=column_names)
    final_result['fitness'] = 0

    for i in range(result.shape[0]):
        final_result.loc[i, 'fitness'] = gp.fitness(result[i])

    print("Iteration: %d times." % gp.iteration)
    print(final_result.loc[final_result['fitness'] == 0])

    final_result.to_csv(os.getcwd() + '/result/' + 'gp-result-total.csv')
    final_result.loc[final_result['fitness'] == 0].to_csv(os.getcwd() + '/result/' + 'gp-result-optimal.csv')


