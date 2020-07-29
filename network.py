import numpy as np
import math

# Ensuring reproducibility
np.random.seed(0)

class NeuralNetwork():
	
	def __init__(self):
		self.input = None
		self.nodes = []
		self.weights = []
		# Regularization strengh
		self.regLossParam = 1e-3


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
		b = np.zeros((1, n))

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


	def dataLoss(self, predResults, trueResults):
		"""
		The data loss function.
		"""
		# L2 loss
		loss = np.square(trueResults - predResults)
		return loss/len(trueResults)


	def regLoss(self):
		"""
		The regularization loss function.
		"""
		if self.regLossParam == 0:
			return 0
		else:
			squaredTotal = 0
			for (W, _) in self.weights:
				squaredTotal += np.sum(np.square(W))

			loss = 0.5 * self.regLossParam * squaredTotal
			return loss


	def lossFunc(self, predResults, trueResults):
		return (self.dataLoss(predResults, trueResults) + self.regLoss())


	def backPropagation(self, trueResults):
		"""
		Function to carry out back-propagation algorithm.
		"""
		predResults = self.getOutput()
		# Step 1: Find the gradient at output
		h = 0.001 * np.ones(predResults.shape)
		doutput = (self.lossFunc(predResults + h, trueResults) - self.lossFunc(predResults - h, trueResults))/(2*h)

		# Step 2: back propagate
		dW = np.dot(self.input.T, doutput)
		db = np.sum(doutput, axis=0, keepdims=True)

		# Parameter update
		# Works only for linear classifier (no hidden layer)
		step_size = 0.001
		(W, b) = self.weights[0]
		W += -step_size * dW
		b += -step_size * db
		self.weights[0] = (W, b)

	def train(self, Y, epochs):
		"""
		Train the neural network.
		"""
		for i in range(epochs):
			self.backPropagation(Y)

	def predict(self, X):
		"""
		Make predictions.
		"""
		self.input = X
		return self.getOutput()