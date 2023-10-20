#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <iostream>
#include <vector>

struct NeuralSwapData
{
	int layerId;
	int size;
	int seed;
};

struct NeuralNetworkData
{
	int nb_input_layer;
	int nb_output_layer;
	int nb_hiden_layer;
	int nb_col_hiden_layer;
	int activationSize;
	int weightSize;
	int select_sub_best_neural;
	float mutation_rate;
	float mutation_multiplayer;
};

class NeuralNetwork 
{
public:
    //NeuralNetwork(int input,int c);
    /*void loadModel(const std::string& modelPath);
    void setInputData(const std::vector<double>& inputData);
    void saveModel(const std::string& modelPath);
    void Propagate();
    void BackPropagate();
    std::vector<double> predict();    */
};

#endif //!__NEURAL_NETWORK__