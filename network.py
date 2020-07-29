import numpy as np
import math

# Ensuring reproducibility
np.random.seed(0)

class NeuralNetwork():
	
	def __init__(self):
		self.input = None
		self.nodes = []
		self.weights = []


	def addInput(self, inputArray):
		"""
		Set the input data.
		"""
		self.input = inputArray
		self.nodes.append(len(inputArray[0]))


	def layer(self, n):
		"""
		Creates a new layer.
		n - the number of nodes in the layer.
		"""
		# Number of nodes in previous layer
		nPrev = self.nodes[-1]

		# Initializing the weights and biases
		W = np.random.randn(nPrev, n) * math.sqrt(2.0/nPrev) # Recommended initialization method
		b = np.random.randn(1, n)

		# Store them as a tuple
		self.nodes.append(n)
		self.weights.append((W, b))


	def activFunc(self, inputArray):
		"""
		The activation function for the neurons in the network.
		"""
		# ReLU activation
		return np.maximum(0, inputArray)


	def hiddenLayerOutput(self, prevOut, W, b):
		"""
		Returns the output of a hidden layer.
		prevOut - Output from the previous layer (np.array)
		W, b = Weight and bias of this layer
		"""
		layerOutput = np.dot(prevOut, W) + b
		return self.activFunc(layerOutput)


	def finalOutput(self, prevOut, W, b):
		"""
		Returns the output of the final layer.
		Similar to the hiddenLayerOutput(), but without 
		the activation function.
		"""
		final_output = np.dot(prevOut, W) + b
		return final_output


	def getOutput(self):
		"""
		Returns the output of the neural network.
		"""
		h = self.input
		# Loop through the hidden layers
		for i in range(len(self.weights) - 1):
			(W, b) = self.weights[i]
			h = self.hiddenLayerOutput(h, W, b)

		# Return the output
		(W, b) = self.weights[-1]
		return self.finalOutput(h, W, b)