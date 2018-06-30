#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include "NeuralNetwork.h"

using namespace std;

vector<vector<float>> data_load(char *file_name)
{
    // declare variables
    string line_1;
    vector<vector<float>> outputs;
    
    // read file
    ifstream inputfile(file_name, ios::in); // open the file
    while(getline(inputfile, line_1))
    {
        istringstream temp_string(line_1);
        vector <float> col_vector;
        string col_element;
        
        // read every column from the line that is seperated by commas
        while(getline(temp_string, col_element,','))
        {
            int temp_value = atoi( col_element.c_str() );
            col_vector.push_back(temp_value);
        }
        outputs.push_back(col_vector);
    }
    inputfile.close();
    
    // reset array's values
    for (int i = 0 ; i < outputs.size() ; ++i)
    {
        for (int j = 1 ; j < outputs[0].size() ; ++j)
        {
            if (outputs[i][j] != 0.0)
            {
                outputs[i][j] = 1.0;
            }
            else
            {
                outputs[i][j] = 0.0;
            }
        }
    }
    
    return outputs;
}

// initialize weight matrix between input layer and hidden layer
float** setting_weight_input_hidden(int hidden_number, int input_number)
{
    // initialize 2D matrix
    float** output_matrix = new float*[hidden_number];
    for (int i = 0 ; i < hidden_number ; i = i + 1)
    {
        output_matrix[i] = new float[input_number];
    }
    
    for (int i = 0 ; i < hidden_number ; i++)
    {
        for (int j = 0 ; j < input_number ; j++)
        {
            int sign = rand() % 2;
            
            output_matrix[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1)
            {
                output_matrix[i][j] = - output_matrix[i][j];
            }
        }
    }
    
    return output_matrix;
}

// initialize weight matrix between hidden layer and output layer
float** setting_weight_hidden_output(int output_number, int hidden_number)
{
    // initialize 2D matrix
    float** output_matrix_2 = new float*[output_number];
    for (int i = 0 ; i < output_number ; i = i + 1)
    {
        output_matrix_2[i] = new float[hidden_number + 1];
    }
    
    for (int i = 0 ; i < output_number ; i++)
    {
        for (int j = 0 ; j < (hidden_number + 1) ; j++)
        {
            int sign = rand() % 2;
            
            output_matrix_2[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1)
            {
                output_matrix_2[i][j] = - output_matrix_2[i][j];
            }
        }
    }
    
    return output_matrix_2;
}

// sigmoid function
float sigmoid_function(float input_value)
{
    input_value = input_value * -1.0;
    return 1.0 / ( 1.0 + exp( input_value ) );
}

// testing stage
void NeuralNetwork::testing_stage(int hidden_number, int output_number)
{
    // testing stage
    // declare variables
    int input_number = 785;
    float* test_hidden_layer = new float[hidden_number];
    float* test_output_layer = new float[output_number];
    int correct_n = 0;
    
    // read testing data
    char test_names[] = "test.csv";
    testing_data_matrix = data_load(test_names);
    
    // read the weight between input and hidden
    vector<vector<float>> weight_input_hidden;
    ifstream inputfile("weight_input_hidden.csv"); // open the file
    int row_number = 0;
    string line_1;
    
    while (getline(inputfile, line_1))
    {
        float temp_value;
        stringstream string_stream_1(line_1);
        weight_input_hidden.push_back( vector<float>() );
        
        while (string_stream_1 >> temp_value)
        {
            weight_input_hidden[row_number].push_back(temp_value);
        }
        
        row_number = row_number + 1;
    }
    inputfile.close();
    
    // read the weight between input and hidden
    vector<vector<float>> weight_hidden_output;
    ifstream inputfile2("weight_hidden_output.csv"); // open the file
    int rows = 0;
    string line_2;
    
    while (getline(inputfile2, line_2))
    {
        float temps;
        stringstream string_stream(line_2);
        weight_hidden_output.push_back( vector<float>() );
        
        while (string_stream >> temps)
        {
            weight_hidden_output[rows].push_back(temps);
        }
        
        rows = rows + 1;
    }
    inputfile2.close();
    
    // testing example loop, count the number of correctly predicted examples in testing data
    for (int testing_example_index = 0 ; testing_example_index < testing_data_matrix.size() ; testing_example_index++)
    {
        
        // compute output value of each testing example
        // computer the output of each hidden node in hidden layer
        for (int hidden_index = 0 ; hidden_index < hidden_number ; hidden_index = hidden_index + 1)
        {
            float temp_value = weight_input_hidden[hidden_index][0] * 1.0;
            for (int index = 1 ; index < 785 ; index = index + 1)
            {
                temp_value = temp_value + ( weight_input_hidden[hidden_index][index] * testing_data_matrix[testing_example_index][index] );
            }
            // compute activation function in hidden layer
            test_hidden_layer[hidden_index] = sigmoid_function(temp_value);
        }
        // compute the output of each output node in output layer
        for (int outpu_index = 0 ; outpu_index < output_number ; outpu_index = outpu_index + 1)
        {
            float temp_value_2 = weight_hidden_output[outpu_index][0] * 1.0;
            for (int index_2 = 1 ; index_2 < (hidden_number + 1) ; index_2 = index_2 + 1)
            {
                temp_value_2 = temp_value_2 + ( weight_hidden_output[outpu_index][index_2] * test_hidden_layer[index_2 - 1] );
            }
            // compute final activation in output layer
            test_output_layer[outpu_index] = sigmoid_function(temp_value_2);
        }
        
        
        // finad maximal value and its position for each training example
        int max_position_output_testing = distance(test_output_layer, max_element(test_output_layer, test_output_layer + 10));
        
        if (max_position_output_testing == testing_data_matrix[testing_example_index][0])
        {
            correct_n++;
        }
    }
    
    cout << "The Correct Number is: " << correct_n << "/" << testing_data_matrix.size() << endl;
    cout << "Classification Performance: " << ((float)correct_n / (float)testing_data_matrix.size()) * 100 << " %" << endl;
}

// execute multi-layer neural network
void NeuralNetwork::execution_function(int hidden_number, int output_number, float learning_rate_value, int max_iter, int mini_size)
{
    cout << "hidden number: " << hidden_number << endl;
    cout << "learning rate: " << learning_rate_value << endl;
    cout << "Maximum Iteration: " << max_iter << endl;
    cout << "The size of mini-batch: " << mini_size << endl;
    
    // setting parameters
    int input_number = 785;
    float* hidden_layer = new float[hidden_number];
    float* output_layer = new float[output_number];
    float* error_hidden_layer = new float[hidden_number];
    float* error_output_layer = new float[output_number];
    
    // initialize array
    for (int i = 0 ; i < hidden_number ; ++i)
    {
        hidden_layer[i] = 0.0;
        error_hidden_layer[i] = 0.0;
    }
    for (int i = 0 ; i < output_number ; ++i)
    {
        output_layer[i] = 0.0;
        error_output_layer[i] = 0.0;
    }
    
    // read training data
    char names[] = "training.csv";
    training_data_matrix = data_load(names);
    
    // initialize target matrix
    float **target_matrix = new float*[training_data_matrix.size()];
    for (int i = 0; i < training_data_matrix.size(); i++)
    {
        target_matrix[i] = new float[output_number];
    }
    // build label matrix, the row of this matrix represents each instance. And each column represents each class type 0 to 9
    for (int i = 0 ; i < training_data_matrix.size() ; i++)
    {
        for (int j = 0 ; j < output_number ; j++)
        {
            target_matrix[i][j] = 0.0;
        }
        target_matrix[i][int(training_data_matrix[i][0])] = 1.0;
    }
    
    // initialize weight matrix between input layer and hidden layer
    float** weight_input_hidden = setting_weight_input_hidden(hidden_number, input_number);
    
    // initialize weight matrix between hidden layer and output layer
    float** weight_hidden_output = setting_weight_hidden_output(output_number, hidden_number);
    
    // declare mini-batch matrix
    float **mini_batch = new float*[mini_size];
    for (int i = 0; i < mini_size ; i++)
    {
        mini_batch[i] = new float[785];
    }
    // initialize array
    for (int i = 0 ; i < mini_size ; ++i)
    {
        for (int j = 0 ; j < 785 ; ++j)
        {
            mini_batch[i][j] = 0.0;
        }
    }
    
    // declare mini-batch label
    float **mini_batch_label = new float*[mini_size];
    for (int i = 0; i < mini_size; i++)
    {
        mini_batch_label[i] = new float[10];
    }
    // initialize array
    for (int i = 0 ; i < mini_size ; ++i)
    {
        for (int j = 0 ; j < 10 ; ++j)
        {
            mini_batch_label[i][j] = 0.0;
        }
    }
    
    // declare temp matrix
    float** temp_hidden_output = new float*[output_number];
    for (int i = 0 ; i < output_number ; i = i + 1)
    {
        temp_hidden_output[i] = new float[hidden_number + 1];
    }
    
    float** temp_input_hidden = new float*[hidden_number];
    for (int i = 0 ; i < hidden_number ; i = i + 1)
    {
        temp_input_hidden[i] = new float[input_number];
    }
    
    // initialize tmp matrix
    for (int i = 0 ; i < output_number ; ++i)
    {
        for (int j = 0 ; j < hidden_number + 1 ; ++j)
        {
            temp_hidden_output[i][j] = 0.0;
        }
    }
    
    for (int i = 0 ; i < hidden_number ; ++i)
    {
        for (int j = 0 ; j < input_number ; ++j)
        {
            temp_input_hidden[i][j] = 0.0;
        }
    }
    
    // neural network algorithm
    // epoch loop
    int n_batch = training_data_matrix.size() / mini_size;
    for (int epoch_index = 0 ; epoch_index < max_iter ; epoch_index = epoch_index + 1)
    {
        
        // mini-batch loop
        for (int batch_index = 1 ; batch_index <= n_batch ; ++batch_index)
        {
            // create mini-batch matrix and mini-batch label
            if (batch_index == 1)
            {
                for (int i = 0 ; i < mini_size ; ++i)
                {
                    for (int j = 0 ; j < 785 ; ++j)
                    {
                        mini_batch[i][j] = training_data_matrix[(batch_index - 1) + i][j];
                    }
                    
                    for (int j = 0 ; j < 10 ; ++j)
                    {
                        mini_batch_label[i][j] = target_matrix[(batch_index - 1) + i][j];
                    }
                }
            }
            else
            {
                for (int i = 0 ; i < mini_size ; ++i)
                {
                    for (int j = 0 ; j < 785 ; ++j)
                    {
                        mini_batch[i][j] = training_data_matrix[((batch_index - 1) + i) + (mini_size - 1)][j];
                    }
                    
                    for (int j = 0 ; j < 10 ; ++j)
                    {
                        mini_batch_label[i][j] = target_matrix[((batch_index - 1) + i) + (mini_size - 1)][j];
                    }
                }
            }
            
            // ============================================
            
            // training example loop, this loop is to check each example and update weights
            for (int example_index = 0 ; example_index < mini_size ; example_index = example_index + 1)
            {
                // First part: forward propagation:
                // forward propagation: computer the output of each hidden node in hidden layer
                for (int hidden_index = 0 ; hidden_index < hidden_number ; hidden_index = hidden_index + 1)
                {
                    float temp_value = weight_input_hidden[hidden_index][0] * 1.0;
                    for (int index = 1 ; index < 785 ; index = index + 1)
                    {
                        temp_value = temp_value + ( weight_input_hidden[hidden_index][index] * mini_batch[example_index][index] );
                    }
                    // compute activation function in hidden layer
                    hidden_layer[hidden_index] = sigmoid_function(temp_value);
                }
                // forward propagation: compute the output of each output node in output layer
                for (int outpu_index = 0 ; outpu_index < output_number ; outpu_index = outpu_index + 1)
                {
                    float temp_value_2 = weight_hidden_output[outpu_index][0] * 1.0;
                    for (int index_2 = 1 ; index_2 < (hidden_number + 1) ; index_2 = index_2 + 1)
                    {
                        temp_value_2 = temp_value_2 + ( weight_hidden_output[outpu_index][index_2] * hidden_layer[index_2 - 1] );
                    }
                    // compute final activation in output layer
                    output_layer[outpu_index] = sigmoid_function(temp_value_2);
                }
                
                // Second part: compute Error value
                // compute error value of each output node in output layer
                for (int i = 0 ; i < output_number ; i++)
                {
                    error_output_layer[i] = output_layer[i] * ( 1.0 - output_layer[i]) * ( mini_batch_label[example_index][i] - output_layer[i] );
                }
                // compute error valuef of each hidden node in hidden layer
                for (int i = 0 ; i < hidden_number ; i++)
                {
                    // compute sum of weights times error values in output layer
                    float temp_value_3 = 0.0;
                    for (int j = 0 ; j < output_number ; j++)
                    {
                        temp_value_3 = temp_value_3 + ( weight_hidden_output[j][i + 1] * error_output_layer[j] );
                    }
                    error_hidden_layer[i] = hidden_layer[i] * (1.0 - hidden_layer[i]) * temp_value_3;
                }
                
                // Third part: back propagation:
                // update weight matrix between hidden layer and output layer
                for (int i = 0 ; i < output_number ; i++)
                {
                    // update weight value
                    temp_hidden_output[i][0] = temp_hidden_output[i][0] + ((learning_rate_value / (float)mini_size) *error_output_layer[i] * 1.0);
                    for (int j = 1 ; j < (hidden_number + 1) ; j++)
                    {
                        // update weight value
                        temp_hidden_output[i][j] = temp_hidden_output[i][j] + ((learning_rate_value / (float)mini_size) *error_output_layer[i] * hidden_layer[j - 1] );
                    }
                }
                
                // update weight matrix between hidden layer and input layer
                for (int i = 0 ; i < hidden_number ; i++)
                {
                    // update weight value
                    temp_input_hidden[i][0] = temp_input_hidden[i][0] + ((learning_rate_value / (float)mini_size) *error_hidden_layer[i] * 1.0 );
                    for (int j = 1 ; j < input_number ; j++)
                    {
                        // update weight value
                        temp_input_hidden[i][j] = temp_input_hidden[i][j] + ((learning_rate_value / (float)mini_size) *error_hidden_layer[i] * mini_batch[example_index][j] );
                    }
                }
                
                // ========================================================================
            } // end example loop
            
            // update weights
            for (int i = 0 ; i < output_number ; i++)
            {
                for (int j = 0 ; j < (hidden_number + 1) ; j++)
                {
                    // update weight value
                    weight_hidden_output[i][j] = weight_hidden_output[i][j] + temp_hidden_output[i][j];
                }
            }
            
            for (int i = 0 ; i < hidden_number ; i++)
            {
                for (int j = 0 ; j < input_number ; j++)
                {
                    // update weight value
                    weight_input_hidden[i][j] = weight_input_hidden[i][j] + temp_input_hidden[i][j];
                    
                }
            }
            
            // set temp matrices to 0
            for (int i = 0 ; i < output_number ; ++i)
            {
                for (int j = 0 ; j < hidden_number + 1 ; ++j)
                {
                    temp_hidden_output[i][j] = 0.0;
                }
            }
            
            for (int i = 0 ; i < hidden_number ; ++i)
            {
                for (int j = 0 ; j < input_number ; ++j)
                {
                    temp_input_hidden[i][j] = 0.0;
                }
            }
        }  // end mini-batch loop
        
        // display each epoch result
        cout << "Epoch: " << epoch_index + 1  << " is done !!" << endl;
        cout << "============================================================" << endl;
        
    }  // end epoch loop
    
    // write the weights between input and hidden
    ofstream ouput1("weight_input_hidden.csv");
    for (int i = 0 ; i < hidden_number ; i++)
    {
        for (int j = 0 ; j < input_number ; j++)
        {
            ouput1 << weight_input_hidden[i][j] <<" ";
        }
        ouput1 << "\n";
    }
    cout << "The Weight has been writen to the file." << endl;
    
    // write the weights between hidden and output
    ofstream ouput2("weight_hidden_output.csv");
    for (int i = 0 ; i < output_number ; i++)
    {
        for (int j = 0 ; j < hidden_number + 1 ; j++)
        {
            ouput2 << weight_hidden_output[i][j] << " ";
        }
        ouput2 << "\n";
    }
    cout << "The Weight has been writen to the file." << endl;
    cout << "Training Stage is done." << endl;
}
