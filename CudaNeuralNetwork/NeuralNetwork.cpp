#include "NeuralNetwork.hpp"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "CNNHelper.hpp"

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

__global__ void PropagateNeuralNetwork(const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, const float* weight_buffer, float* activation_Buffer)
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
			sum += activation_Buffer[i + offsetbaseNN] * weight_buffer[offsetWeight + minOffsetWeight* nnd_Buffer->nb_hiden_layer + i];
		}
		activation_Buffer[index] = Tanh(sum);
	}
	else if (nld->layerId > 1 && nld->layerId < (2 + nnd_Buffer->nb_col_hiden_layer) - 1
		&& index >= nnd_Buffer->nb_input_layer + nnd_Buffer->nb_hiden_layer
		&& index < nnd_Buffer->nb_input_layer + nld->layerId * nnd_Buffer->nb_hiden_layer)
	{
		int offsetbaseNN = (index - (index - nnd_Buffer->nb_input_layer) % nnd_Buffer->nb_hiden_layer)- nnd_Buffer->nb_hiden_layer;
		int offsetWeight = nnd_Buffer->nb_input_layer * nnd_Buffer->nb_hiden_layer + (index - nnd_Buffer->nb_hiden_layer - nnd_Buffer->nb_input_layer) * nnd_Buffer->nb_hiden_layer;
		float sum = 0.0f;
		for (int i = 0; i < nnd_Buffer->nb_hiden_layer; i++)
		{
			sum += activation_Buffer[i + offsetbaseNN] * weight_buffer[i+ offsetWeight];
		}
		activation_Buffer[index] = Tanh(sum);
	}
}

__global__ void BackPropagateNeuralNetwork(const NeuralNetworkData* nnd_Buffer, const NeuralSwapData* nld, const float* weight_buffer, float* activation_Buffer, float* result_Buffer)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= nld->size)
	{
		return;
	}
	

}

NeuralNetwork::NeuralNetwork(float* weight_buffer, float* activation_Buffer, float* result_Buffer, NeuralNetworkData* nnd_Buffer, NeuralSwapData* nld_Buffer, NeuralNetworkData nnd)
{
	m_weight_buffer = weight_buffer;
	m_activation_Buffer = activation_Buffer;
	m_result_Buffer = result_Buffer;
	m_nnd_Buffer = nnd_Buffer;
	m_nld_Buffer = nld_Buffer;
	m_nnd = nnd;
}

NeuralNetwork::~NeuralNetwork()
{
	cudaFree(m_weight_buffer);
	cudaFree(m_activation_Buffer);
	cudaFree(m_result_Buffer);
	cudaFree(m_nnd_Buffer);
	cudaFree(m_nld_Buffer);
}

void NeuralNetwork::loadModel(const std::string& modelPath)
{
	std::cout << "LoadModel : " << modelPath << std::endl;
}

void NeuralNetwork::setInputData(const std::vector<double>& inputData)
{

}

void NeuralNetwork::saveModel(const std::string& modelPath)
{

}

void NeuralNetwork::propagate()
{
	std::cout << "Propagate" << std::endl;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

	int layerStep = 2 + m_nnd_Buffer->nb_col_hiden_layer;
	m_nld_Buffer->size = m_nnd_Buffer->activationSize;
	dim3 dimGrid;
	dim3 dimBlock;

	CNNHelper::KernelDispath(m_nld_Buffer->size, &deviceProp, &dimGrid, &dimBlock);
	for (int i = 0; i < layerStep - 1; i++)
	{
		m_nld_Buffer->layerId = i + 1;
		//PropagateNeuralNetwork<<<dimGrid, dimBlock>>>(m_nnd_Buffer, m_nld_Buffer, m_weight_buffer, m_activation_Buffer);
	}
}

void NeuralNetwork::backPropagate(std::vector<float> prediction_Data)
{
	std::cout << "BackPropagate" << std::endl;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

	cudaStatus = cudaMemcpy(result_Buffer, &prediction_Data.data, sizeof(float)*m_nnd.nb_output_layer, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int layerStep = 2 + m_nnd_Buffer->nb_col_hiden_layer;
	m_nld_Buffer->size = m_nnd_Buffer->activationSize;
	dim3 dimGrid;
	dim3 dimBlock;

	CNNHelper::KernelDispath(m_nld_Buffer->size, &deviceProp, &dimGrid, &dimBlock);
	for (int i = 0; i < layerStep - 1; i++)
	{
		m_nld_Buffer->layerId = i + 1;
		//PropagateNeuralNetwork<<<dimGrid, dimBlock>>>(m_nnd_Buffer, m_nld_Buffer, m_weight_buffer, m_activation_Buffer);
	}
}
