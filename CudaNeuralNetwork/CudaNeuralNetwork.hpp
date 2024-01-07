#ifndef __CUDA_NEURAL_NETWORK__
#define __CUDA_NEURAL_NETWORK__

#include "export.hpp"
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"
#include <string>

class NeuralNetwork;

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd);
typedef NeuralNetwork* (*CreateNeuralNetwork)(NeuralNetworkData nnd);

void trainingNeuralNetwork(NeuralNetwork* nn, const std::string& dataSetPath);
typedef void (*TrainingNeuralNetwork)(NeuralNetwork* nn, const std::string& dataSetPath);

void releaseNeuralNetwork(NeuralNetwork* network);
typedef void(*ReleaseNeuralNetwork)(NeuralNetwork*);

void generateDataSet(const std::string& path, const std::string& dataSetSavepath);
typedef void(*GenerateDataSet)(const std::string&, const std::string&);

int addWithCuda(int* c, const int* a, const int* b, unsigned int size);
typedef int(*AddWithCudaFunc)(int*, const int*, const int*, unsigned int);

#endif //!__CUDA_NEURAL_NETWORK__