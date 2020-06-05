#!/usr/bin/python3
import numpy as np
import sys

#Sample code from https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/
class NeuralNetwork:

	#Constructor for the neural network.
	def __init__(self):
		#Set the seed of the random number generator.
		#np.random.seed(1)

		#Create a 3x2 matrix with random values from (-1, 1).
		self.weights = 2 * np.random.random((2, 2)) - 1

		#Create a 2 array with random values from (-1, 1).
		self.biases = 2 * np.random.random(2) - 1

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
			output = self.predict(inputs)

			#Calculate the cost gradient
			costGradient = np.dot(inputs.T * np.sum(2 * (output - outputs), axis=1), self.sigmoidDerivative(output)) / len(inputs)

			#Adjust the weights by a factor
			self.weights -= costGradient

			costGradientBias = np.dot(np.sum(2 * (output - outputs), axis = 1, keepdims=True).T, self.sigmoidDerivative(output)) / len(inputs)
			self.biases -= costGradientBias.flatten()

	#Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
	def predict(self, inputs):
		return self.sigmoid(np.dot(inputs, self.weights) + self.biases)

#Create neural network object
n = NeuralNetwork()

if len(sys.argv) == 1:
	#Provide the training inputs and outputs
	inputs = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])
	outputs = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])

	#Train the neural network
	n.train(inputs, outputs, 10000)

	#Write all the weights to a file
	f = open("weights02.txt", "w")
	for i in n.weights.flatten():
		f.write(str(i))
		f.write(" ")

	#Write the bias to the file
	f.write("\n")
	for i in n.biases.flatten():
		f.write(str(i))
		f.write(" ")

	f.close()
else:
	#Get file from command-line argument
	filename = sys.argv[1]
	f = open(filename, "r")

	#Get a list of the numbers in the file
	nums = f.read().split()
	z = 0

	for i in range(len(n.weights)):
		for j in range(len(n.weights[i])):
			#Assign the numbers from the file to the weights of the neural network
			n.weights[i,j] = np.float64(nums[z])
			z += 1

	for i in range(len(n.biases)):
		n.biases[i] = np.float64(nums[z])
		z += 1

	#Print out result from examples
	examples = np.array([[0, 0], [1, 0]])
	for example in examples:
		print(example, " might output:", n.predict(example))
