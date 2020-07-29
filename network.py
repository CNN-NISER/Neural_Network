import numpy as np
import math

class NeuralNetwork():
	
	def __init__(self):
		self.input = None
		self.nodes = []
		self.weights = []


	def inputLayer(self, inputArray):
		"""
		Input the first layer.
		"""
		self.input = inputArray
		self.nodes.append(len(inputArray))


	def layer(self, n):
		"""
		Creates a new layer.
		n - the number of nodes in the layer.
		"""
		# Check whether this is an inner layer.
		if self.nodes is not None: 
			# Number of nodes in previous layer
			nPrev = self.nodes[-1]

			# Initializing the weights and biases
			W = np.random.randn(n, nPrev) * math.sqrt(2.0/nPrev) # Recommended initialization method
			b = np.random.randn(n, 1)

			# Store them as a tuple
			self.nodes.append(n)
			self.weights.append((W, b))

		# If this is not an inner layer
		# ask for it first 
		else:
			print("Enter an input layer first by calling the inputLayer() method.")


	def activFunc(self, inputArray):
		"""
		The activation function for the neurons in the network.
		"""
		# ReLU activation
		base = np.zeros(inputArray.shape)
		return np.maximum(base, inputArray)


	def hiddenLayerOutput(self, prevOut, W, b):
		"""
		Returns the output of a hidden layer.
		prevOut - Output from the previous layer (np.array)
		W, b = Weight and bias of this layer
		"""
		layerOutput = np.dot(W, prevOut) + b
		return self.activFunc(layerOutput)


	def finalOutput(self, prevOut, W, b):
		"""
		Returns the output of the final layer.
		Similar to the hiddenLayerOutput(), but without 
		the activation function.
		"""
		final_output = np.dot(W, prevOut) + b
		return final_output


	def getOutput(self):
		"""
		Returns the output of the neural network.
		"""
		(W, b) = self.weights[0]
		h = self.hiddenLayerOutput(self.input, W, b)

		# Loop through the hidden layers
		for i in range(1, len(self.weights) - 1):
			(W, b) = self.weights[i]
			h = self.hiddenLayerOutput(h, W, b)

		# Return the output
		(W, b) = self.weights[-1]
		return self.finalOutput(h, W, b)
