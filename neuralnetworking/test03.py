#!/usr/bin/python3
import numpy as np
import propagationlogic as pl

#Sample code from https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/
class NeuralNetwork:

	#Constructor for the neural network.
	def __init__(self):
		#Set the seed of the random number generator.
		#np.random.seed(1)

		#Create a 3x2 matrix with random values from (-1, 1).
		self.weights1 = 2 * np.random.random((3, 2)) - 1

		#Create a 2x1 matrix with random values from (-1, 1).
		self.weights2 = 2 * np.random.random((2, 1)) - 1

		#Create a 2 array with random values from (-1, 1).
		self.biases1 = 2 * np.random.random(2) - 1

		#Create a 1 array with random values from (-1, 1).
		self.biases2 = 2 * np.random.random(1) - 1

	#Returns the sigmoid function result for a given x.
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	#Returns the sigmoid derivative function result for a given x.
	def sigmoidDerivative(self, x):
		return x * (1 - x)

	#Train the neural network and ajust the weights over time.
	def train(self, inputs, outputs, training_iterations):
		#Loop for however many iterations
		for iteration in range(training_iterations):
			#Pass the training set through the network
			hidden = self.predict(inputs, self.weights1, self.biases1)
			output = self.predict(hidden, self.weights2, self.biases2)

			#Calculate the cost gradient
			costGradient2 = pl.calculateCost(self.weights2, hidden, outputs, output, self.sigmoidDerivative)

			#Adjust the weights by a factor
			self.weights2 -= costGradient2

			costGradientBiases2 = pl.calculateCostBias(self.biases2, hidden, outputs, output, self.sigmoidDerivative)

			self.biases -= costGradientBias2

			costGradient1 = pl.calculateCost(self.weights1, inputs, outputs, outupt, self.sigmoidDerivative)

	#Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
	def predict(self, inputs, weights, biases):
		return self.sigmoid(np.dot(inputs, weights) + biases)

#Create neural network object
n = NeuralNetwork()

#Provide the training inputs and outputs
inputs = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]])
outputs = np.array([[1, 1], [0, 1], [0, 0], [1, 0], [0, 0]])

#Train the neural network
n.train(inputs, outputs, 10000)

#Print out result from examples
examples = np.array([[0, 1, 1], [0, 0, 1]])
for example in examples:
    print(example, " might output:", n.predict(example))
