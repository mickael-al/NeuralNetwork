//Peer Programming: Guo, Albarello
#include "kernel.h"
#include <iostream>

__global__ void addKernel(int* c, const int* a, const int* b,const int * size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size[0])
    {
        return;
    }
    c[i] = a[i] + b[i];
}

__device__ float Tanh(float x)
{
	//return std::tanh(x);
	if (x > 20.0)
	{
		return 1.0;
	}
	else if (x < -20.0)
	{
		return -1.0;
	}
	else
	{
		float exp2x = exp(2 * x);
		return (exp2x - 1) / (exp2x + 1);		
	}
}

__device__ float TanhDerive(float x)
{
	return 1.0f - (x*x);
}

__global__ void propagateNeuralNetworkCompact(const NeuralSwapData* nld, const NeuralNetworkDataCompact* nndc,const int * self_d, float*** self_w, float** self_x)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j >= nld->size || j == 0)
	{
		return;
	}
	float sum = 0.0f;
	for (int i = 0; i < self_d[nld->l-1]+1; i++)
	{
		sum += self_w[nld->l][i][j] * self_x[nld->l -1][i];
	}
	if (nld->l < nndc->self_l || nndc->is_classification)
	{
		sum = Tanh(sum);
	}
	self_x[nld->l][j] = sum;
}

__global__ void backPropagateNeuralNetworkCompact(const NeuralSwapData* nld,const int* self_d, float*** self_w, float** self_x, float** self_delta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nld->size || i == 0)
	{
		return;
	}
	float sum = 0.0f;
	for (int j = 1; j < self_d[nld->l] + 1; j++)
	{
		sum += self_w[nld->l][i][j] * self_delta[nld->l][j];
	}
	self_delta[nld->l - 1][i] = TanhDerive(self_x[nld->l -1][i])* sum;
}

__global__ void backPropagateNeuralNetworkCompactEnd(const NeuralSwapData* nld, const NeuralNetworkDataCompact* nndc,const int* self_d, float*** self_w, float** self_x, float** self_delta)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= nld->size)
	{
		return;
	}
	for (int j = 1; j < self_d[nld->l] + 1; j++)
	{
		self_w[nld->l][i][j] -= nndc->alpha * self_x[nld->l - 1][i] * self_delta[nld->l][j];
	}
}

void AddKernel(dim3 dimGrid, dim3 dimBlock, int* c, const int* a, const int* b, const int* size)
{
    addKernel<<<dimGrid, dimBlock>>>(c, a, b, size);
}

void PropagateNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, const NeuralNetworkDataCompact* nndc, const int* self_d, float*** self_w, float** self_x)
{
	propagateNeuralNetworkCompact <<<dimGrid, dimBlock >>> (nld, nndc, self_d, self_w,self_x);
}

void BackPropagateNeuralNetworkCompact(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, const int* self_d, float*** self_w,  float** self_x, float** self_delta)
{
	backPropagateNeuralNetworkCompact <<<dimGrid, dimBlock >>> (nld, self_d, self_w, self_x, self_delta);
}

void BackPropagateNeuralNetworkCompactEnd(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, const NeuralNetworkDataCompact* nndc,const int* self_d, float*** self_w, float** self_x, float** self_delta)
{
	backPropagateNeuralNetworkCompactEnd <<<dimGrid, dimBlock >>> (nld, nndc, self_d, self_w, self_x, self_delta);
}
