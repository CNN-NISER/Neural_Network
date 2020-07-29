import numpy as np

class NeuralNetwork():
	
	#	def __init__(self):
	#	self.input = None
	#	self.nodes = []
	#	self.weights = []


	#def inputLayer(self, inputArray):
	#	"""
	#	Input the first layer.
	#	"""
	#	self.input = inputArray

	def __init__(self):
		a=[0]  #Sample
		self.nodes = []
		self.weights = []
		self.nodes.append(len(a))
		self.input = []
		self.input.append(a[0])

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
			W = np.random.randn(n, nPrev)  #Random numbers (Small random numbers do not perform well compared to this)
			b = np.zeros([n,1])  #In the future when memory and arithmetic time becomes an issue, use bias trick

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


	def getOutput(self, inp):
		"""
		Returns the output of the neural network.
		"""
		output = []
		for i in range(0,len(inp)):
			self.input = np.array(inp[i])
			(W, b) = self.weights[0]
			h = self.hiddenLayerOutput(self.input, W, b)
			# Loop through the hidden layers
			for i in range(1, len(self.weights) - 1):
				(W, b) = self.weights[i]
				h = self.hiddenLayerOutput(h, W, b)
				# Return the output
				(W, b) = self.weights[-1]
				output.append(self.finalOutput(h, W, b))
		return output

	def Loss(self, y1, y2):
		l2=0
		for i in range(0,len(y1)):
			l2 += y2[i] - y1[i]
		l2 = l2/len(y1)
		return l2
		

	def training(self, inputData, expValue, epoch, learn, delta):
		for i in range(0, epoch):
			#print('Epoch number: ',i)
			if(i%10 == 0):
				learn=learn/2
				#print('Learning rate updated to: ', learn)
			"""Update w for diff, then find derivative, then subtract dW"""
			for j in range(len(self.weights)):
				l1 = self.Loss(self.getOutput(inputData), expValue)
				#print(l1)
				(W,b) = self.weights[j]
				numrows = len(W)
				numcols = len(W[0])
				for k in range(numrows):
					for l in range(numcols):
						W[k][l] += delta
						self.weights[j] = (W,b)
						l2 = self.Loss(self.getOutput(inputData), expValue)
						m = (l2 - l1)/delta
						#dw = (m*W[k][l] - l2)/m
						if m!=0:
							W[k][l] -= m*learn
						W[k][l] -= delta
						self.weights[j] = (W,b)
		print('Training process complete.')



