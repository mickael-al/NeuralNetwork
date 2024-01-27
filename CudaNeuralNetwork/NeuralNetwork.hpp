#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <iostream>
#include <vector>
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"

class NeuralNetwork
{
public:
	NeuralNetwork(NeuralNetworkData nnd);
	~NeuralNetwork();
	void trainingDataSet(const std::string& dataSetPath);
	void loadModel(const std::string& modelPath);
	void setInputData(const std::vector<double>& inputData);
	void saveModel(const std::string& modelPath);
	void propagate();
	void backPropagate(std::vector<float> prediction_Data);
private:
	//gpu
	float *** m_self_w;
	float ** m_self_x;
	float ** m_self_delta;
	int* m_self_d;
	NeuralSwapData* m_nld_Buffer;
	NeuralNetworkDataCompact* m_nndc_Buffer;
	//gpu

	NeuralNetworkData m_nnd;
	NeuralSwapData m_nld;
	NeuralNetworkDataCompact m_nndc;
	int* m_array_size_d;
	float* m_outDelta;
	float* m_activation;

};

#endif //!__NEURAL_NETWORK__