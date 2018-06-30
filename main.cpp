/*
<Li-Yun> <Wang>
<11-10-2016>
CS 541
Lab 3
*/

#include <iostream>
#include "NeuralNetwork.h"

using namespace std;

// main function
int main(int argc, char* argv[])
{
    // setting parameters
    int hidden_node_number;
    int outout_node_number = 10;
    float learning_rate;
    int itera_n;
    int mini_size;
    
    // check argument format
    if (argc == 1)
    {
        cout << "Program Error." << endl;
        cout << "This program must have two arguments: (1) ./output and (2) run or testing" << endl;
        cout << "Please try again." << endl;
    }
    else
    {
        // get character arracy, and convert char array to string
        string input_string(argv[1]);
        
        // Input format is correct
        if ( input_string.compare("run") == 0 )
        {
            // get the number of hidden node and output node
            hidden_node_number = atoi(argv[2]);
            
            // get learning rate and alhpa value
            learning_rate = atof(argv[3]);
            itera_n = atoi(argv[4]);
            mini_size = atoi(argv[5]);
            
            // call the function of multi-layer neural network
            NeuralNetwork* neural_network = new NeuralNetwork();
            neural_network -> execution_function(hidden_node_number, outout_node_number, learning_rate, itera_n, mini_size);
        }
        else if ( input_string.compare("testing") == 0 )
        {
            // get the number of hidden node and output node
            hidden_node_number = atoi(argv[2]);
            
            // call the function of multi-layer neural network
            NeuralNetwork* neural_network = new NeuralNetwork();
            neural_network -> testing_stage(hidden_node_number, outout_node_number);
        }
        else
        {
            cout << "Input Name is Error." << endl;
            cout << "Input Name is run_neural_network, and" << endl;
            cout << "Please Enter hyper-parameters: the number of hidden nodes, learning rate, and momentum rate." << endl;
        }
    }
    
    return 0;
}
