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


	def getLayerOutput(self, n):
		"""
		Returns the output of the nth layer of the neural network.
		n = 0 is the input layer
		0 <= n <= len(self.weights)
		"""
		penLayer = len(self.weights) - 1 # The penultimate layer
		h = self.input

		# Loop through the hidden layers
		for i in range(min(n, penLayer)):
			(W, b) = self.weights[i]
			h = self.hiddenLayerOutput(h, W, b)

		# Return the output
		if n <= penLayer:
			return h
		else:
			(W, b) = self.weights[n-1]
			return self.finalOutput(h, W, b)


	def dataLoss(self, predResults, trueResults):
		"""
		Returns the data loss.
		"""
		# L2 loss
		loss = np.square(trueResults - predResults)
		return loss/len(trueResults)


	def regLoss(self):
		"""
		Returns the regularization loss.
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
		Updates weights by carrying out backpropagation.
		"""
		predResults = self.getLayerOutput(len(self.weights))
		
		h = 0.001 * np.ones(predResults.shape)
		step_size = 1e-5

		doutput = (self.lossFunc(predResults + h, trueResults) - self.lossFunc(predResults - h, trueResults))/(2*h)
		
		# Weights
		nPrev = len(self.weights)

		while nPrev - 1 >= 0:

			if nPrev != len(self.weights):
				# Backprop into hidden layer
				dhidden = np.dot(doutput, W.T)
				# Backprop the ReLU non-linearity
				dhidden[prevLayer <= 0] = 0
			else:
				dhidden = doutput

			nPrev += -1

			prevLayer = self.getLayerOutput(nPrev)
			dW = np.dot(prevLayer.T, dhidden)
			db = np.sum(dhidden, axis=0, keepdims=True)
			(W, b) = self.weights[nPrev]
			W += -step_size * dW
			b += -step_size * db
			self.weights[nPrev] = (W, b)

			doutput = dhidden

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
		return self.getLayerOutput(len(self.weights))
