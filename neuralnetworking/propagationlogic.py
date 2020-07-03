import numpy as np

def sigmoidDerivative(x):
	return x * (1 - x)

def calculateCost(weights, inputs, outputs, output, activationDerivative):
	costGradient = np.zeros(weights.shape)

	for i in range(len(weights)):
		for j in range(len(weights[i])):
			tsum = 0
			for k in range(len(inputs)):
				changeinztow = inputs[k,i]
				changeinatoz = activationDerivative(output[k,j])

				changeinctoz = 0
				for l in range(len(weights[i])):
					changeinctoz += 2 * (output[k,l] - outputs[k,l])

				changeinctow = changeinztow * changeinatoz * changeinctoz
				tsum += changeinctow

			costGradient[i,j] = tsum / len(inputs)
	return costGradient

def calculateBiasCost(biases, inputs, outputs, output, activationDerivative):
	biasCostGradient = np.zeros(biases.shape)

	for i in range(len(biases)):
		tsum = 0
		for j in range(len(inputs)):
			changeinbtow = 1
			changeinatoz = activationDerivative(output[j,i])

			changeinctoz = 0
			for k in range(len(biases)):
				changeinctoz += 2 * (output[j,k] - outputs[j,k])

			changeinctow = changeinbtow * changeinatoz * changeinctoz
			tsum += changeinctow

		biasCostGradient[i] = tsum / len(inputs)
	return biasCostGradient
