#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define TRAINING 4
#define INPUT 3
#define OUTPUT 1

float** createArray(int n, int m) {
	float* values = calloc(m*n, sizeof(float));
	float** rows = malloc(n*sizeof(float*));

	for(int i = 0; i < n; i++) {
		rows[i] = values + i*m;
	}

	return rows;
}

//Learns by getting the sum of the inputs times their respective weights and then returning the sigmoid result of it.
float** dot(float a[TRAINING][INPUT], float b[INPUT][OUTPUT]) {
	float** output;
	output = createArray(TRAINING, OUTPUT);

	for(int i = 0; i < TRAINING; i++) {
		for(int j = 0; j < OUTPUT; j++) {
			int sum = 0;

			for(int k = 0; k < INPUT; k++) {
				sum += a[i][k] * b[k][j];
			}
			output[i][j] = sum;
		}
	}
	return output;
}

int main() {
	float inputs[TRAINING][INPUT] = {{0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {0, 1, 0}};
	float outputs[INPUT][OUTPUT]  = {{1}, {2}, {3}};

	float** product;
	product = dot(inputs, outputs);

	for(int i = 0; i < TRAINING; i++) {
		for(int j = 0; j < OUTPUT; j++) {
			printf("%f ", product[i][j]);
		}
		printf("\n");
	}

	free(product);

	return 0;
}
