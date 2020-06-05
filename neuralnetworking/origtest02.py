#!/usr/bin/python3
from numpy import *

#Sample code from https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/
class NeuralNetwork:

	#Constructor for the neural network.
	def __init__(self):
		#Set the seed of the random number generator.
		random.seed(1)

		#Create a 3x2 matrix with random values from (-1, 1).
		self.weights1 = 2 * random.random((3, 2)) - 1
		self.weights2 = 2 * random.random((2, 1)) - 1

		self.biases1

	#Returns the sigmoid function result for a given x.
	def sigmoid(self, x):
		return 1 / (1 + exp(-x))

	#Returns the sigmoid derivative function result for a given x.
	def sigmoidDerivative(self, x):
		return x * (1 - x)

	#Train the neural network and ajust the weights over time.
	def train(self, inputs, outputs, training_iterations):
		#Loop for however many iterations
		for iteration in range(training_iterations):
			#Pass the training set through the network
			hidden = self.predict(inputs, self.weights1)

			#Pass the training set through the network
			output = self.predict(hidden, self.weights2)

			#Calculate the cost value
			costArray = (outputs - output)**2
			costValue = sum(costArray, 1)

			#Adjust the weights by a factor
			self.weights1 += -GradC of W + self.weights1

	#Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
	def predict(self, inputs, weights):
		#TODO: add biases
		return self.sigmoid(dot(inputs, weights))

#Create neural network object
n = NeuralNetwork()

#Provide the training inputs and outputs
inputs = array([[0, 1, 1], [0, 0, 1], [1, 0, 0], [1, 1, 0]])
outputs = array([[1], [1], [0], [0]])

#Train the neural network
n.train(inputs, outputs, 10000)

#Print out result from examples
examples = array([[1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1]])
for example in examples:
    print(example, " might output:", n.predict(example))
