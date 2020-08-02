# Neural_Network

This repository contains code for a neural network (without the use of any packages, apart from NumPy) that can multiply any number by 2 (and can be changed to any multiplication). The code was made in a way to inculcate any number of hidden layers and nodes, although 0 or 1 would suffice for multiplication. Of course, this can be extended to solve (using Neural Networks) any problem using any number of input and output nodes.

<p align="center">
  <img src="/images/test.png"><br>
  <imgcaption>Training and prediction</imgcaption>
</p>


<p align="center">
  <img src="/images/result.png"><br>
  <imgcaption>Results</imgcaption>
</p>

We initialised the weights by random numbers normalised by the number of nodes in the previous layer. Biases are initialised to 0. 

We use the Rectified Linear Unit for the activation function and a simple L2 norm loss with weight regularisation as our loss function, which in itself, gives pretty good results. Since we use numerical differentiation, the L2 loss that was used can be switched to any other loss function.

Further optimisation, hyper-parameter optimisation and generalisation (for different activation functions) will be done along the way.
