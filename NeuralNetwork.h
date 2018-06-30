#ifndef __program__NeuralNetwork__
#define __program__NeuralNetwork__

#include <vector>

using namespace std;

class NeuralNetwork
{
    
private:
    vector< vector<float> > training_data_matrix;
    vector< vector<float> > testing_data_matrix;
public:
    NeuralNetwork(){};
    ~NeuralNetwork(){};
    
    // execution function
    void execution_function(int hidden_number, int output_number, float learning_rate_value, int max_iter, int mini_size);
    
    // testing function
    void testing_stage(int hidden_number, int output_number);
};

#endif