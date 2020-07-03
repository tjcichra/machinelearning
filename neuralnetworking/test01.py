#!/usr/bin/python3
import numpy as np
import propagationlogic as pl
import sys

#Sample code from https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/
class NeuralNetwork:

	#Constructor for the neural network.
	def __init__(self):
		#Set the seed of the random number generator.
		#np.random.seed(1)

		#Create a 3x1 matrix with random values from (-1, 1).
		self.weights = 2 * np.random.random((3, 1)) - 1
		self.bias = 2 * np.random.random(1) - 1

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
			#costGradient = np.dot(inputs.T * 2 * (output - outputs).flatten(), self.sigmoidDerivative(output)) / len(inputs)
			costGradient = pl.calculateCost(self.weights, inputs, outputs, output, self.sigmoidDerivative)

			#Readjust the weights
			self.weights -= costGradient

			#Calculate the cost gradient for the bias
			costBias = pl.calculateBiasCost(self.bias, inputs, outputs, output, self.sigmoidDerivative)
			self.bias -= costBias
			#costGradientBiasd = np.dot(2 * (output - outputs).T, self.sigmoidDerivative(output))
			#costGradientBias = costGradientBias / len(inputs)
			#self.bias -= costGradientBias[0,0]

	#Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
	def predict(self, inputs):
		return self.sigmoid(np.dot(inputs, self.weights) + self.bias)

	def calculateCost(self, inputs, outputs, output):
		costGradient = np.zeros(self.weights.shape)

		for i in range(len(self.weights)):
			for j in range(len(self.weights[i])):
				tsum = 0
				for k in range(len(inputs)):
					changeinztow = inputs[k,i]
					changeinatoz = self.sigmoidDerivative(output[k,0])
					changeinctoz = 2 * (output[k,0] - outputs[k,0])

					changeinctow = changeinztow * changeinatoz * changeinctoz
					tsum += changeinctow

				costGradient[i,j] = tsum / len(inputs)
		return costGradient

	def calculateBiasCost(self, inputs, outputs, output):
		costBias = np.zeros(self.bias.shape)

		for i in range(len(self.bias)):
			tsum = 0
			for k in range(len(inputs)):
				changeinztob = 1
				changeinatoz = self.sigmoidDerivative(output[k,0])
				changeinctoa = 2 * (output[k,0] - outputs[k,0])

				changeinctob = changeinztob * changeinatoz * changeinctoa
				tsum += changeinctob
			costBias[i] = tsum / len(inputs)
		return costBias

#Create neural network object
n = NeuralNetwork()

if len(sys.argv) == 1:
	#Provide the training inputs and outputs
	inputs = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1]])
	outputs = np.array([[1], [0], [1]])

	#Train the neural network
	n.train(inputs, outputs, 10000)

	#Write all the weights to a file
	f = open("weights01.txt", "w")
	for i in n.weights.flatten():
		f.write(str(i))
		f.write(" ")

	#Write the bias to the file
	f.write("\n")
	f.write(str(n.bias[0]))

	#Close file
	f.close()
else:
	#Get file from command-line argument
	filename = sys.argv[1]
	f = open(filename, "r")

	#Get a list of the numbers in the file
	nums = f.read().split()
	z = 0

	#Loop through the weights
	for i in range(len(n.weights)):
		for j in range(len(n.weights[i])):
			#Assign the numbers from the file to the weights of the neural network
			n.weights[i,j] = np.float64(nums[z])
			z += 1

	#Assign the number from the file to the bias of the neural network
	n.bias = np.float64(nums[z])

	#Close the file
	f.close()

	#Print out result from examples
	examples = np.array([[0, 1, 1], [0, 0, 1], [1, 1, 1], [0, 1, 0], [0, 0, 0]])
	for example in examples:
   		print(example, " might output:", n.predict(example))
