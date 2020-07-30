# Neural_Network

This repository contains code for a neural network (without the use of any packages, apart from NumPy) that can multiply any number by 2 (and can be changed to any multiplication). The code was made in a way to inculcate any number of hidden layers and nodes in every layer, although 0 or just 1 would suffice for multiplication. Ofcourse this can be extended to solve (using Neural Networks) any problem using any number of input and output nodes.

We initialsed the weights by random numbers normalised by the number of nodes in the previous layer. Biases are initialised to 0. 

We use the Rectified Linear Unit for the activation function and a simple L1 norm loss with weight regularisation as our loss function, which in itself, gives pretty good results.

Further optimisation, hyper-parameter optimisation and generalisation (for activation and loss functions) will be done along the way.
