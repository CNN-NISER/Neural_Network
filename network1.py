# This is an attempt to shorten and simplify the previous code.

import numpy as np
import random

class NeuralNetwork():

	def __init__(self, ln_arr):
		self.ln_arr = ln_arr # the array defining the no.of layers and nodes
		self.nlayer = len(ln_arr) # no.of layers in the neural network

		# initialising the biases; no need for biases in the input layer
		self.biases = [np.random.randn(y, 1) for y in ln_arr[1:]]
		# creating a matrix of weights for the all the neurons
		self.weights = [np.random.randn(y, x) for x, y in zip(ln_arr[:-1], ln_arr[1:])]


	def activFunc(self, inputArray):
		"""
		The activation function for the neurons in the network.
		"""
		# ReLU activation
		base = np.zeros(inputArray.shape)
		return np.maximum(base, inputArray)

	def relu_prime(self, z):
		
		y = z
		y[y > 0] = 1
		y[y <= 0] = 0

		return y



	def getOutput(self, inputArray):
		"""
		This function returns the output for a given input to the
		neural network.
		"""

		for bias, wght in zip(self.biases, self.weights):
			inputArray = self.activFunc(np.dot(wght, inputArray)+bias)

		return inputArray

	def GradDesc(self, train_data, epochs, mini_batch_size, learn_rate, test_data=None):

		if test_data:
			n_test = len(test_data)
		n_train = len(train_data)

		for i in range(epochs):

			random.shuffle(train_data)
			mini_batches = [ train_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]
			
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch , learn_rate)

			if test_data:
				print("Epoch {0}: {1} ".format(i, self.evaluate(test_data)))
			else:
					print("Epoch {0} complete".format(i))



	def update_mini_batch(self , mini_batch , learn_rate):

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		for x, y in mini_batch:
			delta_nabla_b , delta_nabla_w = self.backprop(x, y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b , delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w , delta_nabla_w)]
		self.weights = [w-(learn_rate/len(mini_batch))*nw for w, nw in zip(self.weights , nabla_w)]
		self.biases = [b-(learn_rate/len(mini_batch))*nb for b, nb in zip(self.biases , nabla_b)]

	def backprop(self, x, y):

		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		# feedforward

		activation = x
		activations = [x]
		zs = []

		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation) + b
			zs.append(z)

			activation = self.activFunc(z)
			activations.append(activation)

		# backward pass

		delta = self.cost_derivative(activations[-1], y) * self.relu_prime(zs[-1]) # NOTE THIS LINE

		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta , activations[-2].transpose())


		for l in range(2, self.nlayer):
			z = zs[-l]

			# NOTE THIS LINE
			rp = self.relu_prime(z)

			delta = np.dot(self.weights[-l+1].transpose(), delta) * rp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta , activations[-l-1].transpose())

		return (nabla_b , nabla_w)


	def evaluate(self, test_data):

		
		test_results = np.array([self.getOutput(x) for (x, y) in test_data]) # stores the test results
		rmse = np.sqrt(np.mean(np.array([(a-y)**2 for a,(x,y) in zip(test_results, test_data)])))

		return rmse

	def cost_derivative(self, output_activations, y):
		
		return (output_activations - y)

































