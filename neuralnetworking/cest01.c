//Sample code from https://www.geeksforgeeks.org/implementing-ann-training-process-in-python/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#define INPUT 3
#define OUTPUT 1
#define TRAINING 3
#define EXAMPLE 5

int weights[INPUT][OUTPUT];

//Constructor for the neural network.
void init() {
	//Set the seed of the random number generator.
	srand(time(NULL));

	//Fill the 3x1 matrix of weights with random values from (-1, 1).
	for(int i = 0; i < INPUT; i++) {
		weights[i][0] = 2 * (rand() / (double) RAND_MAX) - 1;
	}
}

//Returns the sigmoid function result for a given x.
void sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

//Returns the sigmoid derivative function result for a given x.
void sigmoidDerivative(double x) {
	return x * (1 - x)
}

//Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
void predict(double inputs[TRAINING][INPUT]) {
	double output[TRAINING][OUTPUT];

	for(int i = 0; i < TRAINING; i++) {
		for(int j = 0; j < OUTPUT; j++) {
			int sum = 0;

			for(int k = 0; k < INPUT; k++) {
				sum += inputs[i, k] * weights[k, j];
			}
			output[i, j] = sum
		}
	}
	return self.sigmoid(dot(inputs, self.weights))
}

//Train the neural network and ajust the weights over time.
void train(double inputs[TRAINING][INPUT], double outputs[TRAINING][OUTPUT], int training_iterations) {
	//Loop for however many iterations
	for(int x = 0; x < training_iteration; x++) {
		//Pass the training set through the network
		output = predict(inputs)

		//Calculate the error
		error = outputs - output

		//Adjust the weights by a factor
		factor = dot(inputs.T, error * self.sigmoidDerivative(output))
		self.weights += factor
	}
}

void main() {
	//Provide the training inputs and outputs
	double inputs[TRAINING][INPUT] = {{0, 1, 1}, {1, 0, 0}, {1, 0, 1}}
	double outputs[TRAINING][OUTPUT] = {{1}, {0}, {1}}

	//Train the neural network
	train(inputs, outputs, 10000);

	//Print out result from examples
	double examples[EXAMPLE][INPUT] = {{1, 0, 1}, {1, 1, 1}, {0, 1, 0}, {1, 1, 1}, {1, 1, 0}}

	for example in examples {
		print(example, " might output:", n.predict(example))
	}
}
