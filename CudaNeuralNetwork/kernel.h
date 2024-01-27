#ifndef KERNEL_H
#define KERNEL_H

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"

void AddKernel(dim3 dimGrid, dim3 dimBlock,int* c, const int* a, const int* b, const int* size);
void InitNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, float* weight_buffer);
void PropagateNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, float* weight_buffer, float* activation_Buffer);
void BackPropagateNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, float* weight_buffer, float* activation_Buffer, float* delta_Buffer, float* result_Buffer);

#endif // KERNEL_H