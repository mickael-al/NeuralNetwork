#ifndef KERNEL_H
#define KERNEL_H

#include "NeuralNetwork.hpp"

NeuralNetwork* createNeuralNetwork();
void releaseNeuralNetwork(NeuralNetwork* network);
int addWithCuda(int* c, const int* a, const int* b, unsigned int size);
typedef int(*AddWithCudaFunc)(int*, const int*, const int*, unsigned int);

#endif // KERNEL_H