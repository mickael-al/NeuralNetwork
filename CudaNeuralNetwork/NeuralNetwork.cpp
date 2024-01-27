#include "NeuralNetwork.hpp"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include "CNNHelper.hpp"
#include "kernel.h"
#include <cstdlib>
#include <ctime>

NeuralNetwork::NeuralNetwork(NeuralNetworkData nnd)
{
	nnd.activationSize = (nnd.nb_input_layer+1) + (nnd.nb_col_hiden_layer) * (nnd.nb_hiden_layer+1) + (nnd.nb_output_layer+1);
	nnd.weightSize = (nnd.nb_input_layer+1) * (nnd.nb_hiden_layer+1);
	for (int i = 0; i < nnd.nb_col_hiden_layer - 1; i++)
	{
		nnd.weightSize += (nnd.nb_hiden_layer + 1) * (nnd.nb_hiden_layer + 1);
	}
	nnd.weightSize += (nnd.nb_hiden_layer + 1) * (nnd.nb_output_layer + 1);
	int layerStep = 2 + nnd.nb_col_hiden_layer;
	m_nld.size = nnd.weightSize;
	m_nld.seed = 0;
	m_array_size_d = new int[nnd.nb_col_hiden_layer + 2];
	m_array_size_d[0] = nnd.nb_input_layer;
	for (int i = 0; i < nnd.nb_col_hiden_layer; i++)
	{
		m_array_size_d[i+1] = nnd.nb_hiden_layer;
	}
	m_array_size_d[nnd.nb_col_hiden_layer + 1] = nnd.nb_output_layer;
	m_nndc.self_l = nnd.nb_col_hiden_layer + 1;
	m_nndc.alpha = nnd.alpha;
	m_nndc.is_classification = nnd.is_classification;
	std::srand(static_cast<unsigned int>(std::time(0)));	

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);

	cudaMallocManaged((void****)&m_self_w, (nnd.nb_col_hiden_layer + 2) * sizeof(float**));
	for (int l = 1; l < nnd.nb_col_hiden_layer + 2; l++)
	{		
		cudaMallocManaged((void***)&m_self_w[l], (m_array_size_d[l - 1] + 1) * sizeof(float*));
		for (int i = 0; i < (m_array_size_d[l - 1] + 1); i++)
		{
			cudaMallocManaged((void**)&m_self_w[l][i], (m_array_size_d[l] + 1) * sizeof(float));
			for (int j = 0; j < (m_array_size_d[l] + 1); j++)
			{
				if (j == 0)
				{
					m_self_w[l][i][j] = 1.0f;
				}
				else
				{
					m_self_w[l][i][j] = ((static_cast<float>(rand()) / RAND_MAX) * 2.0f) - 1;
				}
			}
		}
	}
	cudaDeviceSynchronize();
	cudaMallocManaged((void***)&m_self_x, (nnd.nb_col_hiden_layer + 2) * sizeof(float*));
	cudaMallocManaged((void***)&m_self_delta, (nnd.nb_col_hiden_layer + 2) * sizeof(float*));

	for (int l = 0; l < nnd.nb_col_hiden_layer + 2; l++)
	{
		cudaMallocManaged((void**)&m_self_x[l], (m_array_size_d[l] + 1) * sizeof(float));
		cudaMallocManaged((void**)&m_self_delta[l], (m_array_size_d[l] + 1) * sizeof(float));
		for (int j = 0; j < (m_array_size_d[l] + 1); j++)
		{
			m_self_delta[l][j] = 0.0f;
			if (j == 0)
			{
				m_self_x[l][j] = 1.0f;
			}
			else
			{
				m_self_x[l][j] = 0.0f;
			}
		}
	}
	cudaDeviceSynchronize();
	cudaMalloc((void**)&m_self_d, (nnd.nb_col_hiden_layer + 2) * sizeof(int));
	cudaMemcpy(m_self_d, m_array_size_d, (nnd.nb_col_hiden_layer + 2) * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&m_nndc_Buffer, sizeof(NeuralNetworkDataCompact));
	cudaMemcpy(m_nndc_Buffer, &m_nndc, sizeof(NeuralNetworkDataCompact), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&m_nld_Buffer, sizeof(NeuralSwapData));
	cudaMemcpy(m_nld_Buffer, &m_nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);

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

	m_outDelta = new float[m_array_size_d[m_nndc.self_l] + 1];
	m_activation = new float[m_array_size_d[m_nndc.self_l] + 1];

	m_nnd = nnd;
}

NeuralNetwork::~NeuralNetwork()
{
	delete m_outDelta;
	delete m_activation;	
	for (int l = 1; l < m_nnd.nb_col_hiden_layer + 2; l++)
	{
		for (int i = 0; i < (m_array_size_d[l - 1] + 1); i++)
		{
			cudaFree(m_self_w[l][i]);
		}

		cudaFree(m_self_w[l]);
	}
	cudaFree(m_self_w);

	for (int l = 0; l < m_nnd.nb_col_hiden_layer + 2; l++)
	{
		cudaFree(m_self_x[l]);
		cudaFree(m_self_delta[l]);
	}

	cudaFree(m_self_x);
	cudaFree(m_self_delta);

	delete m_array_size_d;
	cudaFree(m_nld_Buffer);
	cudaFree(m_nndc_Buffer);	
}

void NeuralNetwork::trainingInput(const std::vector<std::vector<float>> input, const std::vector<std::vector<float>> output,float min_percent_error_train)
{
	if (input.size() != output.size())
	{
		fprintf(stderr, "input size : %d not equal to output size : %d", input.size(), output.size());
		return;
	}
	for (int i = 0; i < input.size(); i++)
	{
		if (input[i].size() != m_nnd.nb_input_layer)
		{
			fprintf(stderr, "input[%d] size : %d not equal to input layer size : %d",i, input[i].size(), m_nnd.nb_input_layer);
			return;
		}
		if (output[i].size() != m_nnd.nb_output_layer)
		{
			fprintf(stderr, "output[%d] size : %d not equal to output layer size : %d", i, output[i].size(), m_nnd.nb_output_layer);
			return;
		}
	}
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	float* result_compare = new float[m_nnd.nb_output_layer];
	float errormoy = 0.0f;
	int gardeFou = 100001;
	for (int j = 0; j < gardeFou; j++)
	{
		errormoy = 0.0f;
		for (int i = 0; i < input.size(); i++)
		{
			cudaMemcpy(m_self_x[0] + 1, input[i].data(), sizeof(float) * m_nnd.nb_input_layer, cudaMemcpyHostToDevice);
			propagate();
			backPropagate(output[i]);
			cudaMemcpy(result_compare, m_self_x[m_nndc.self_l] + 1, sizeof(float) * m_nnd.nb_output_layer, cudaMemcpyDeviceToHost);
			for (int l = 0; l < m_nnd.nb_output_layer; l++)
			{
				errormoy += abs(output[i][l] - result_compare[l]) * (1.0f/ m_nnd.nb_output_layer);
			}
		}
		errormoy = errormoy / input.size();
		std::cout << "Error: " << errormoy * 100.0f << "%" << std::endl;
		if (errormoy*100.0f < min_percent_error_train)
		{
			j = gardeFou;
		}
	}
	delete[] result_compare;
}

void NeuralNetwork::trainingDataSet(const std::map<const std::string, std::vector<float*>> & data,int input_size, float min_percent_error_train)
{
	if (data.size() != m_nnd.nb_output_layer)
	{
		fprintf(stderr, "data size : %d not equal to output size : %d", data.size(), m_nnd.nb_output_layer);
		return;
	}
	if (input_size != m_nnd.nb_input_layer)
	{
		fprintf(stderr, "input_size image : %d not equal to input layer size : %d", input_size, m_nnd.nb_input_layer);
		return;
	}
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return;
	}

	cudaDeviceProp deviceProp;
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	float errormoy = 0.0f;
	int gardeFou = 100001;
	float* result_compare = new float[m_nnd.nb_output_layer];
	std::vector<std::vector<float>> output;
	int totalDivideError = 0;
	for (std::pair<const std::string, std::vector<float*>> d : data)
	{
		totalDivideError+=d.second.size();
	}
	for (int i = 0; i < data.size(); i++)
	{
		std::vector<float> co;		
		for (int j = 0; j < data.size(); j++)
		{
			co.push_back(i == j ? 1 : 0);
		}
		output.push_back(co);
	}
	for (int j = 0; j < gardeFou; j++)
	{		
		errormoy = 0.0f;
		int countData = 0;
		for (std::pair<const std::string, std::vector<float*>> d : data)
		{			
			int currentd = 0;
			for (float* ptr : d.second) 
			{				
				cudaMemcpy(m_self_x[0] + 1, ptr, sizeof(float) * m_nnd.nb_input_layer, cudaMemcpyHostToDevice);
				propagate();
				backPropagate(output[countData]);
				cudaMemcpy(result_compare, m_self_x[m_nndc.self_l] + 1, sizeof(float) * m_nnd.nb_output_layer, cudaMemcpyDeviceToHost);
				for (int l = 0; l < m_nnd.nb_output_layer; l++)
				{
					errormoy += abs(output[countData][l] - result_compare[l]) * (1.0f/ m_nnd.nb_output_layer);					
				}
				if (currentd % 100 == 0)
				{
					std::cout << "Image : " << currentd << " Errormoy : " << errormoy << std::endl;
				}
				currentd++;
			}
			countData++;
		}	
		errormoy = errormoy / totalDivideError;
		std::cout << "Error: " << errormoy * 100.0f << "%" << std::endl;
		if (errormoy * 100.0f < min_percent_error_train)
		{
			j = gardeFou;
			delete[] result_compare;
			return;
		}
	}
	delete[] result_compare;
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

	dim3 dimGrid;
	dim3 dimBlock;	

	for (int l = 1; l < m_nnd.nb_col_hiden_layer + 2;l++)
	{
		m_nld.size = m_array_size_d[l] + 1;
		m_nld.l = l;
		CNNHelper::KernelDispath(m_array_size_d[l]+1, &deviceProp, &dimGrid, &dimBlock);
		cudaMemcpy(m_nld_Buffer, &m_nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);		
		PropagateNeuralNetwork(dimGrid,dimBlock, m_nld_Buffer, m_nndc_Buffer, m_self_d, m_self_w, m_self_x);
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
	
	cudaMemcpy(m_activation, m_self_x[m_nndc.self_l], sizeof(float)* (m_array_size_d[m_nndc.self_l] + 1), cudaMemcpyDeviceToHost);
	for (int j = 1; j < m_array_size_d[m_nndc.self_l] + 1; j++)
	{
		m_outDelta[j] = m_activation[j] - prediction_Data[j - 1];
		if (m_nndc.is_classification)
		{
			m_outDelta[j] *= 1.0f - (m_activation[j]* m_activation[j]);
		}
	}
	cudaMemcpy(m_self_delta[m_nndc.self_l]+1, m_outDelta+1, sizeof(float)* m_array_size_d[m_nndc.self_l], cudaMemcpyHostToDevice);

	dim3 dimGrid;
	dim3 dimBlock;

	for (int l = m_nndc.self_l; l >= 2; l--)
	{
		m_nld.size = m_array_size_d[l - 1] + 1;
		m_nld.l = l;
		CNNHelper::KernelDispath(m_nld.size, &deviceProp, &dimGrid, &dimBlock);
		cudaMemcpy(m_nld_Buffer, &m_nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);
		BackPropagateNeuralNetworkCompact(dimGrid, dimBlock, m_nld_Buffer, m_self_d, m_self_w, m_self_x, m_self_delta);
		cudaDeviceSynchronize();
	}

	for (int l = 1; l < m_nndc.self_l+1; l++)
	{
		m_nld.size = m_array_size_d[l - 1] + 1;
		m_nld.l = l;
		CNNHelper::KernelDispath(m_nld.size, &deviceProp, &dimGrid, &dimBlock);
		cudaMemcpy(m_nld_Buffer, &m_nld, sizeof(NeuralSwapData), cudaMemcpyHostToDevice);
		BackPropagateNeuralNetworkCompactEnd(dimGrid, dimBlock, m_nld_Buffer, m_nndc_Buffer, m_self_d, m_self_w, m_self_x, m_self_delta);
		cudaDeviceSynchronize();
	}
	//std::cout << std::endl;
}