#!/usr/bin/python3
from mnist import MNIST
from numpy import *
import sys

#Sample code from https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/
class NeuralNetwork:

	#Constructor for the neural network.
	def __init__(self):
		#Set the seed of the random number generator.
		#random.seed(1)

		#Create a 784x10 matrix with random values from (-1, 1).
		self.weights = (2 * random.random((784, 10))) - 1

	#Returns the sigmoid function result for a given x.
	def tanh(self, x):
		return tanh(x)

	#Returns the sigmoid derivative function result for a given x.
	def tanhDerivative(self, x):
		return x * (1 - x)

	#Train the neural network and ajust the weights over time.
	def train(self, inputs, outputs, training_iterations):
		#Loop for however many iterations
		for iteration in range(training_iterations):
			#Pass the training set through the network
			output = self.predict(inputs)
			print(output)

			#Calculate the error
			error = outputs - output

			#Adjust the weights by a factor
			factor = dot(inputs.T, error * self.tanhDerivative(output))
			self.weights += factor

	#Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
	def predict(self, inputs):
		return self.tanh(dot(inputs, self.weights))

#Create neural network object
n = NeuralNetwork()

mndata = MNIST("samples")

if len(sys.argv) == 1:
	#Load the training samples
	images, labels = mndata.load_training()
	print("Done loading the training samples.")

	#Caluclate inputs
	inputs = array(images, dtype=longdouble)
	inputs = inputs / 255
	print("Done adjusting the inputs.")

	#Calculate outputs
	outputs = zeros(shape=(len(labels), 10), dtype=longdouble)

	for index, num in enumerate(labels):
		outputs[index, num] = 1

	#Train the neural network
	n.train(inputs, outputs, 100)

	f = open("weights04.txt", "w")
	for i in n.weights.flat:
		f.write(str(i))
		f.write(" ")
	f.close()

	print("Done with training")
else:
	filename = sys.argv[1]

	f = open(filename, "r")
	numbers = f.read().split(' ')
	z = 0

	for i in range(784):
		for j in range(10):
			n.weights[i][j] = longdouble(numbers[z])
			z += 1
	
	f.close()

	images, labels = mndata.load_testing()
	print("Done loading the testing samples.")
	inputs = array(images, dtype=longdouble)
	inputs = inputs / 255
	print("Done adjusting the inputs.")
	print(inputs)

	#for index, image in enumerate(inputs):
	output = n.predict(inputs)
	print(output)
