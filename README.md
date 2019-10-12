## genetic-programming
The project illustrates how to train a TLU by genetic programming.

## Environment
- python 3.0 or above
- pandas, numpy

## Execution
- python gp.py

## Introduction
The project has designed and implemented a genetic programming system to evolve some perceptrons that match well with a given training set. A training set is a collection of tuples of the form (x1, ..., xn, l), where xi’s are real numbers and l is either 1 (positive example) or 0 (negative example).

### Individual
For the genetic programming system, an “individual” is just a tuple (w1,...,wn,θ) of numbers (weights and the threshold).

### Fitness function
The fitness function is the sum of squared errors where fitness∈[0,M]. The “individual” which has less value of fitness is fitter.

### copy operator
Copy 10% of the programs from generation n to generation n+1. These individuals are chosen by the following tournament selection process: 10 individuals are randomly selected from the generation n, and the most fit of these ten is chosen.

### crossover operator
The 89% are produced by generation n by a crossover operation as follows: select a mother and a father from generation n by the tournament selection process, and interchange w_j of the father with w_j of the mother by randomly choosing j,j∈[0,n],j∈Z^+ .

### mutation operator
The 1% are produced by generation n by a mutation operation as follows: select a member from generation n by the tournament selection process, and then a randomly chosen w_j of the member is deleted and replaced by a random number.

### iteration criterial
When there is at least one indiviudal whose fitness function’s value is equal to 0, I would stop the evolution.