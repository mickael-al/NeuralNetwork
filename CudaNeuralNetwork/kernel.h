#ifndef KERNEL_H
#define KERNEL_H

#include "NeuralNetwork.hpp"

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd);
typedef NeuralNetwork(*CreateNeuralNetwork)(NeuralNetworkData nnd);
void releaseNeuralNetwork(NeuralNetwork* network);
int addWithCuda(int* c, const int* a, const int* b, unsigned int size);
typedef int(*AddWithCudaFunc)(int*, const int*, const int*, unsigned int);

#endif // KERNEL_H