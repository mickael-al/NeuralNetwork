#include "NeuralNetwork.hpp"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "CNNHelper.hpp"
#include "kernel.h"

NeuralNetwork::NeuralNetwork(NeuralNetworkData nnd)
{
	nnd.nb_input_layer++;
	nnd.nb_hiden_layer++;
	nnd.nb_output_layer++;
	nnd.activationSize = nnd.nb_input_layer + nnd.nb_col_hiden_layer * nnd.nb_hiden_layer + nnd.nb_output_layer;
	nnd.weightSize = nnd.nb_input_layer * nnd.nb_hiden_layer;
	for (int i = 0; i < nnd.nb_col_hiden_layer - 1; i++)
	{
		nnd.weightSize += nnd.nb_hiden_layer * nnd.nb_hiden_layer;
	}
	nnd.weightSize += nnd.nb_hiden_layer * nnd.nb_output_layer;
	int layerStep = 2 + nnd.nb_col_hiden_layer;
	m_nld.size = nnd.weightSize;
	m_nld.seed = 0;
	NeuralNetworkData* nnd_Buffer = 0;
	NeuralSwapData* nld_Buffer = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

	cudaStatus = cudaMalloc((void**)&m_weight_buffer, nnd.weightSize * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&m_activation_Buffer, nnd.activationSize * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&m_result_Buffer, nnd.nb_output_layer * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&m_delta_Buffer, nnd.activationSize * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}	

	cudaStatus = cudaMalloc((void**)&nnd_Buffer, sizeof(NeuralNetworkData));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMalloc((void**)&nld_Buffer, sizeof(NeuralSwapData));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return;
	}

	cudaStatus = cudaMemcpy(nnd_Buffer, &nnd, sizeof(NeuralNetworkData), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return;
	}

	cudaStatus = cudaMemcpy(nld_Buffer, &m_nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return;
	}
	dim3 dimGrid;
	dim3 dimBlock;

	CNNHelper::KernelDispath(m_nld.size, &deviceProp, &dimGrid, &dimBlock);
	InitNeuralNetwork(dimGrid, dimBlock,nld_Buffer, m_weight_buffer);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "InitNeuralNetwork launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return;
	}

	std::cout << nnd.weightSize << " " << nnd.activationSize << std::endl;

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
	cudaFree(m_delta_Buffer);
}

void NeuralNetwork::trainingDataSet(const std::string& dataSetPath)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	std::vector<std::vector<float>> xor_data;
	xor_data.push_back({ 1,1,1 });
	xor_data.push_back({ 1,-1,1 });
	xor_data.push_back({ -1,1,1 });
	xor_data.push_back({ -1,-1,1 });
	std::vector<std::vector<float>> xor_result_data;
	xor_result_data.push_back({ -1,1 });
	xor_result_data.push_back({ 1,1 });
	xor_result_data.push_back({ 1,1 });
	xor_result_data.push_back({ 1,1 });
	float* result_compare = new float[1];
	float* re = new float[m_nnd.activationSize];
	float errormoy = 0.0f;
	for (int j = 0; j < 100001; j++)
	{
		errormoy = 0.0f;
		for (int i = 0; i < 4; i++)
		{
			cudaMemcpy(m_activation_Buffer, xor_data[i].data(), sizeof(float) * m_nnd.nb_input_layer, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			propagate();
			backPropagate(xor_result_data[i]);
			//cudaMemcpy(result_compare, m_activation_Buffer+(m_nnd.activationSize- m_nnd.nb_output_layer), sizeof(float) * m_nnd.nb_output_layer, cudaMemcpyDeviceToHost);
			//errormoy += abs(xor_result_data[i][0] - result_compare[0]);		
			cudaDeviceSynchronize();
			cudaMemcpy(re, m_activation_Buffer, sizeof(float) * m_nnd.activationSize, cudaMemcpyDeviceToHost);
			if (j % 100 == 0)
			{
				for (int i = 0; i < m_nnd.activationSize; i++)
				{
					std::cout << re[i] << " ,";
				}
				std::cout << std::endl;
			}
		}
		if (j % 100 == 0)
		{
			std::cout << std::endl;
		}
		//errormoy = errormoy / 4.0f;
		//std::cout << errormoy << std::endl;
	}
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
	//std::cout << "Propagate" << std::endl;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

	int layerStep = 2 + m_nnd.nb_col_hiden_layer;
	m_nld.size = m_nnd.activationSize;
	dim3 dimGrid;
	dim3 dimBlock;

	CNNHelper::KernelDispath(m_nld.size, &deviceProp, &dimGrid, &dimBlock);
	for (int i = 0; i < layerStep - 1; i++)
	{
		m_nld.layerId = i + 1;
		cudaMemcpy(m_nld_Buffer, &m_nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		PropagateNeuralNetwork(dimGrid, dimBlock,m_nnd_Buffer, m_nld_Buffer, m_weight_buffer, m_activation_Buffer);
		cudaDeviceSynchronize();
	}
}

void NeuralNetwork::backPropagate(std::vector<float> prediction_Data)
{
	//std::cout << "BackPropagate" << std::endl;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

	cudaStatus = cudaMemcpy(m_result_Buffer, prediction_Data.data(), sizeof(float) * m_nnd.nb_output_layer, cudaMemcpyHostToDevice);
	
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return;
	}

	int layerStep = 2 + m_nnd.nb_col_hiden_layer;
	m_nld.size = m_nnd.activationSize;
	dim3 dimGrid;
	dim3 dimBlock;

	CNNHelper::KernelDispath(m_nld.size, &deviceProp, &dimGrid, &dimBlock);
	for (int i = layerStep - 2; i >= -1; i--)
	{
		m_nld.layerId = i + 1;
		//std::cout << m_nld.layerId << " ,";
		cudaMemcpy(m_nld_Buffer, &m_nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		BackPropagateNeuralNetwork(dimGrid, dimBlock,m_nnd_Buffer, m_nld_Buffer, m_weight_buffer, m_activation_Buffer, m_delta_Buffer, m_result_Buffer);
		cudaDeviceSynchronize();
	}
	//std::cout << std::endl;
}