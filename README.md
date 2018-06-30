# MNIST

This digit recognition program is written by C++, and it is a simple multilayer neural network
classification system that uses mini-batch gradient descent to execute learning process during
training stage. Please follow the instructions to compile the program and execute the program: <br />

1. Compile The Program: <br />
before running the digit classification program, we have to compile the program correctly. Please
use the following command to compile this program: <br />

g++ -std=c++11 main.cpp NeuralNetwork.cpp -o file_name <br />

For example: <br />

g++ -std=c++11 main.cpp NeuralNetwork.cpp -o output <br />

2. Execute The Program: <br />
after compiling the program, we are able to apply two different commands to execute different functions. 
In the folder, there are two pre-trained weights, so users can directly utilize these two weight’s models 
to test classification accuracy. Alternatively, users can run training function in the program to generate 
new weights and apply new weights to do testing stage. <br />

2.1: The command for training neural network: <br />

./output run [N_neurons LR max_iter size_mini-batch] <br />

For example: <br />
./output run 128 0.3 30 10 <br />

2.2: The command for testing neural network: <br />

./output testing [N_neurons] <br />

Ex: <br />
./output testing 128 <br />

where N_neurons:  the number of neurons in the hidden layer. <br />
      LR:  learning rate. <br />
      max_iter:  maximum iteration <br />
      size_mini-batch:  the size of mini-batch <br />

