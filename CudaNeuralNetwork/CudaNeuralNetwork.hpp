#ifndef __CUDA_NEURAL_NETWORK__
#define __CUDA_NEURAL_NETWORK__

#include "export.hpp"
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"
#include <string>
#include <vector>

class NeuralNetwork;

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd);
typedef NeuralNetwork* (*CreateNeuralNetwork)(NeuralNetworkData nnd);

void trainingNeuralNetwork(NeuralNetwork* nn, const std::string& dataSetPath, float min_percent_error_train);
typedef void (*TrainingNeuralNetwork)(NeuralNetwork*, const std::string&, float);

void trainingNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<float>> input, const std::vector<std::vector<float>> output, float min_percent_error_train);
typedef void (*TrainingNeuralNetworkInput)(NeuralNetwork*, const std::vector<std::vector<float>>, const std::vector<std::vector<float>>, float);

void releaseNeuralNetwork(NeuralNetwork* network);
typedef void(*ReleaseNeuralNetwork)(NeuralNetwork*);

void generateDataSet(const std::string& path, const std::string& dataSetSavepath,int image_data_size);
typedef void(*GenerateDataSet)(const std::string&, const std::string&, int);

int addWithCuda(int* c, const int* a, const int* b, unsigned int size);
typedef int(*AddWithCudaFunc)(int*, const int*, const int*, unsigned int);

#endif //!__CUDA_NEURAL_NETWORK__