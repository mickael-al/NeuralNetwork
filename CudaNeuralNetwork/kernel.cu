
#include "kernel.h"

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

	float tan = Tanh(x);
	return 1.0 - tan * tan;
}

__global__ void initNeuralNetwork(const NeuralSwapData* nld, float* weight_buffer)
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


__global__ void propagateNeuralNetwork(const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, const float* weight_buffer, float* activation_Buffer)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nld->size)
	{
		return;
	}
	if (index >= nnd_Buffer->nb_input_layer && index < nnd_Buffer->nb_input_layer + nnd_Buffer->nb_hiden_layer && nld->layerId == 1)//layerId = 1
	{
		float sum = 0.0f;
		for (int i = 0; i < nnd_Buffer->nb_input_layer; i++)
		{
			sum += activation_Buffer[i] * weight_buffer[((index - nnd_Buffer->nb_input_layer) * nnd_Buffer->nb_input_layer) + i];
		}
		activation_Buffer[index] = Tanh(sum);
	}
	else if (index >= (nnd_Buffer->activationSize - nnd_Buffer->nb_output_layer) && nld->layerId == (2 + nnd_Buffer->nb_col_hiden_layer) - 1)
	{
		int offsetbaseNN = nnd_Buffer->nb_input_layer + (nld->layerId - 2) * nnd_Buffer->nb_hiden_layer;
		int offsetWeight = nnd_Buffer->nb_input_layer * nnd_Buffer->nb_hiden_layer + (nld->layerId - 2) * nnd_Buffer->nb_hiden_layer * nnd_Buffer->nb_hiden_layer;
		int minOffsetWeight = index - (nnd_Buffer->activationSize - nnd_Buffer->nb_output_layer);
		float sum = 0.0f;
		for (int i = 0; i < nnd_Buffer->nb_hiden_layer; i++)
		{
			sum += activation_Buffer[i + offsetbaseNN] * weight_buffer[offsetWeight + minOffsetWeight * nnd_Buffer->nb_hiden_layer + i];
		}
		activation_Buffer[index] = Tanh(sum);
	}
	else if (nld->layerId > 1 && nld->layerId < (2 + nnd_Buffer->nb_col_hiden_layer) - 1
		&& index >= nnd_Buffer->nb_input_layer + nnd_Buffer->nb_hiden_layer
		&& index < nnd_Buffer->nb_input_layer + nld->layerId * nnd_Buffer->nb_hiden_layer)
	{
		int offsetbaseNN = (index - (index - nnd_Buffer->nb_input_layer) % nnd_Buffer->nb_hiden_layer) - nnd_Buffer->nb_hiden_layer;
		int offsetWeight = nnd_Buffer->nb_input_layer * nnd_Buffer->nb_hiden_layer + (index - nnd_Buffer->nb_hiden_layer - nnd_Buffer->nb_input_layer) * nnd_Buffer->nb_hiden_layer;
		float sum = 0.0f;
		for (int i = 0; i < nnd_Buffer->nb_hiden_layer; i++)
		{
			sum += activation_Buffer[i + offsetbaseNN] * weight_buffer[i + offsetWeight];
		}
		activation_Buffer[index] = Tanh(sum);
	}
}

__global__ void backPropagateNeuralNetwork(const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, float* weight_buffer, float* activation_Buffer, float* delta_Buffer, float* result_Buffer)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nld->size)
	{
		return;
	}

	if (index >= (nnd_Buffer->activationSize - nnd_Buffer->nb_output_layer) && nld->layerId == (2 + nnd_Buffer->nb_col_hiden_layer) - 1)
	{
		int offsetbaseNN = nnd_Buffer->nb_input_layer + (nld->layerId - 1) * nnd_Buffer->nb_hiden_layer;
		delta_Buffer[index] = (1 - (activation_Buffer[index] * activation_Buffer[index])) * (activation_Buffer[index] - result_Buffer[index - offsetbaseNN]);
	}
	else if (nld->layerId == nnd_Buffer->nb_col_hiden_layer &&
		index >= nnd_Buffer->nb_input_layer + (nld->layerId - 1) * nnd_Buffer->nb_hiden_layer &&
		index < nnd_Buffer->nb_input_layer + nld->layerId * nnd_Buffer->nb_hiden_layer)
	{
		int offsetDelta = nnd_Buffer->nb_input_layer + nld->layerId * nnd_Buffer->nb_hiden_layer;
		int offsetWeight = nnd_Buffer->nb_input_layer * nnd_Buffer->nb_hiden_layer + (nld->layerId - 1) * nnd_Buffer->nb_hiden_layer * nnd_Buffer->nb_hiden_layer;
		int minOffsetWeight = (index - ((nnd_Buffer->activationSize - nnd_Buffer->nb_hiden_layer) - nnd_Buffer->nb_output_layer)) * nnd_Buffer->nb_output_layer;
		float sum = 0.0f;
		for (int i = 0; i < nnd_Buffer->nb_output_layer; i++)
		{
			sum += delta_Buffer[offsetDelta + index] * weight_buffer[offsetWeight + minOffsetWeight + i];
		}
		delta_Buffer[index] = (1 - (activation_Buffer[index] * activation_Buffer[index])) * sum;
		for (int i = 0; i < nnd_Buffer->nb_output_layer; i++)
		{
			weight_buffer[offsetWeight + minOffsetWeight + i] -= nnd_Buffer->mutation_multiplayer * activation_Buffer[index] * delta_Buffer[index];
		}
	}
	else if (nld->layerId > 1 && nld->layerId < nnd_Buffer->nb_col_hiden_layer
		&& index >= nnd_Buffer->nb_input_layer
		&& index < nnd_Buffer->nb_input_layer + nld->layerId * nnd_Buffer->nb_hiden_layer)
	{
		int nblayer = 1 + nnd_Buffer->nb_col_hiden_layer;
		int offsetDelta = nnd_Buffer->nb_input_layer + nld->layerId * nnd_Buffer->nb_hiden_layer;
		int offsetWeight = nnd_Buffer->nb_input_layer * nnd_Buffer->nb_hiden_layer + (nld->layerId - 1) * nnd_Buffer->nb_hiden_layer * nnd_Buffer->nb_hiden_layer;
		int minOffsetWeight = (index - ((nnd_Buffer->activationSize - (nblayer - nld->layerId) * nnd_Buffer->nb_hiden_layer) - nnd_Buffer->nb_output_layer)) * nnd_Buffer->nb_hiden_layer;
		float sum = 0.0f;
		for (int i = 0; i < nnd_Buffer->nb_hiden_layer; i++)
		{
			sum += delta_Buffer[offsetDelta + index] * weight_buffer[offsetWeight + minOffsetWeight + i];
		}
		delta_Buffer[index] = (1 - (activation_Buffer[index] * activation_Buffer[index])) * sum;
		for (int i = 0; i < nnd_Buffer->nb_hiden_layer; i++)
		{
			weight_buffer[offsetWeight + minOffsetWeight + i] -= nnd_Buffer->mutation_multiplayer * activation_Buffer[index] * delta_Buffer[index];
		}
	}
	else if (index < nnd_Buffer->nb_input_layer && nld->layerId == 0)
	{
		int nblayer = 1 + nnd_Buffer->nb_col_hiden_layer;
		int offsetDelta = nnd_Buffer->nb_input_layer;
		int offsetWeight = index * nnd_Buffer->nb_hiden_layer;
		float sum = 0.0f;
		for (int i = 0; i < nnd_Buffer->nb_hiden_layer; i++)
		{
			sum += delta_Buffer[offsetDelta + index] * weight_buffer[offsetWeight + i];
		}
		delta_Buffer[index] = (1 - (activation_Buffer[index] * activation_Buffer[index])) * sum;
		for (int i = 0; i < nnd_Buffer->nb_hiden_layer; i++)
		{
			weight_buffer[offsetWeight + i] -= nnd_Buffer->mutation_multiplayer * activation_Buffer[index] * delta_Buffer[index];
		}
	}
}

void AddKernel(dim3 dimGrid, dim3 dimBlock, int* c, const int* a, const int* b, const int* size)
{
    addKernel<<<dimGrid, dimBlock>>>(c, a, b, size);
}

void InitNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralSwapData* nld, float* weight_buffer)
{
	initNeuralNetwork<<<dimGrid, dimBlock>>>(nld, weight_buffer);
}

void PropagateNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, const float* weight_buffer, float* activation_Buffer)
{
	propagateNeuralNetwork<<<dimGrid, dimBlock>>>(nnd_Buffer, nld, weight_buffer, activation_Buffer);
}

void BackPropagateNeuralNetwork(dim3 dimGrid, dim3 dimBlock, const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, float* weight_buffer, float* activation_Buffer, float* delta_Buffer, float* result_Buffer)
{
	backPropagateNeuralNetwork<<<dimGrid, dimBlock>>>(nnd_Buffer, nld, weight_buffer, activation_Buffer, delta_Buffer, result_Buffer);
}