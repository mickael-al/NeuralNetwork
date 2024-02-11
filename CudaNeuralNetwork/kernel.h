//Peer Programming: Guo, Albarello
#ifndef KERNEL_H
#define KERNEL_H

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"

void AddKernel(dim3 dimGrid, dim3 dimBlock,int* c, const int* a, const int* b, const int* size);
void PropagateNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, const NeuralNetworkDataCompact* nndc, const int* self_d, double*** self_w, double** self_x);
void BackPropagateNeuralNetworkCompact(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, const int* self_d, double*** self_w, double** self_x, double** self_delta);
void BackPropagateNeuralNetworkCompactEnd(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, const NeuralNetworkDataCompact* nndc, const int* self_d, double*** self_w, double** self_x, double** self_delta);

#endif // KERNEL_H