
#include "kernel.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "CNNHelper.hpp"
#include <stdio.h>
#include "NeuralNetwork.hpp"

__global__ void InitNeuralNetwork(const NeuralSwapData * nld, float* weight_buffer)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nld->size)
    {
        return;
    }
    curandState state;
    curand_init(nld->seed, i, 0, &state);
    weight_buffer[i] = curand_uniform(&state) * 2.0f - 1.0f;
}

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd)
{
    NeuralSwapData nld{};

    if (nnd.nb_input_layer <= 0 || nnd.nb_col_hiden_layer <= 0 || nnd.nb_hiden_layer <= 0 || nnd.nb_output_layer <= 0)
    {
        fprintf(stderr, "createNeuralNetwork failed! invalid input NeuralNetworkData\n");
        return nullptr;
    }

    nnd.activationSize = nnd.nb_input_layer + nnd.nb_col_hiden_layer * nnd.nb_hiden_layer + nnd.nb_output_layer;
    nnd.weightSize = nnd.nb_input_layer * nnd.nb_hiden_layer;
    for (int i = 0; i < nnd.nb_col_hiden_layer - 1; i++)
    {
        nnd.weightSize += nnd.nb_hiden_layer * nnd.nb_hiden_layer;
    }
    nnd.weightSize += nnd.nb_hiden_layer * nnd.nb_output_layer;
    int layerStep = 2 + nnd.nb_col_hiden_layer;
    nld.size = nnd.weightSize;
    float* weight_buffer = 0;
    float * activation_Buffer = 0;
    NeuralNetworkData* nnd_Buffer = 0;
    NeuralSwapData* nld_Buffer = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

    cudaStatus = cudaMalloc((void**)&weight_buffer, nnd.weightSize * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&activation_Buffer, nnd.activationSize * sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&nnd_Buffer, sizeof(NeuralNetworkData));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&nld_Buffer, sizeof(NeuralSwapData));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(nnd_Buffer, &nnd, sizeof(NeuralNetworkData), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(nld_Buffer, &nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    dim3 dimGrid;
    dim3 dimBlock;
    
    CNNHelper::KernelDispath(nld.size, &deviceProp, &dimGrid, &dimBlock);
    InitNeuralNetwork<<<dimGrid, dimBlock>>>(nld_Buffer, weight_buffer);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "InitNeuralNetwork launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    return new NeuralNetwork(weight_buffer, activation_Buffer, nnd_Buffer, nld_Buffer);

Error:
    cudaFree(weight_buffer);
    cudaFree(activation_Buffer);
    cudaFree(nnd_Buffer);
    cudaFree(nld_Buffer);

    return nullptr;
}

void releaseNeuralNetwork(NeuralNetwork* network)
{
    delete network;
}

__global__ void addKernel(int* c, const int* a, const int* b,const int * size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size[0])
    {
        return;
    }
    c[i] = a[i] + b[i];
}

int addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int* thread_size = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceProperties failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
  
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&thread_size, sizeof(int));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(thread_size, &size, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 dimGrid;
    dim3 dimBlock;

    CNNHelper::KernelDispath(size, &deviceProp, &dimGrid, &dimBlock);
    addKernel <<<dimGrid, dimBlock >>> (dev_c, dev_a, dev_b, thread_size);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
