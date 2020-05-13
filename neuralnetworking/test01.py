#!/usr/bin/python3
from numpy import *

#Sample code from https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/
class NeuralNetwork:

	#Constructor for the neural network.
	def __init__(self):
		#Set the seed of the random number generator.
		#random.seed(1)

		#Create a 3x1 matrix with random values from (-1, 1).
		self.weights = 2 * random.random((3, 1)) - 1

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
			output = self.predict(inputs)

			#Calculate the error
			error = outputs - output

			#Adjust the weights by a factor
			factor = dot(inputs.T, error * self.sigmoidDerivative(output))
			self.weights += factor

	#Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
	def predict(self, inputs):
		return self.sigmoid(dot(inputs, self.weights))

#Create neural network object
n = NeuralNetwork()

#Provide the training inputs and outputs
inputs = array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
outputs = array([[1], [0], [1]])

#Train the neural network
n.train(inputs, outputs, 10000)

#Print out result from examples
examples = array([[1, 0, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1], [1, 1, 0]])
for example in examples:
    print(example, " might output:", n.predict(example))
