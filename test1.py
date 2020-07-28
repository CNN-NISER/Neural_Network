import numpy as np

from network1 import NeuralNetwork


# Create instance of NeuralNetwork
layer_node = [1,10,10,10,1] # input the number of nodes in the layers as a list
model = NeuralNetwork(layer_node)


# Generating random integers for training the data

low = 1
high = 50000
size = 40000

inp_num = np.random.randint(low,high,size) #creates an array of the nos for training
train_X = np.array(inp_num)
train_X.reshape(size,1)

train_Y = np.array(inp_num*3)
train_Y.reshape(size,1)

# The training data should be a list of tuples (x,y) where x and y are both 1-d arrays
# x is the input and y is the desired output.
train_data = list(zip(train_X,train_Y))


# Training the model
n_epoch = 30
learning_rate = 3
mini_batch_size = 100
model.GradDesc(train_data, n_epoch, mini_batch_size, learning_rate, test_data=None)
