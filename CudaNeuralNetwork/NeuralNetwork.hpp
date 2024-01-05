#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <iostream>
#include <vector>
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"

class NeuralNetwork
{
public:
	NeuralNetwork(float* weight_buffer, float* activation_Buffer, float* result_Buffer, NeuralNetworkData* nnd_Buffer, NeuralSwapData* nld_Buffer, NeuralNetworkData nnd);
	~NeuralNetwork();
	void loadModel(const std::string& modelPath);
	void setInputData(const std::vector<double>& inputData);
	void saveModel(const std::string& modelPath);
	void propagate();
	void backPropagate(std::vector<float> prediction_Data);
private:
	float* m_weight_buffer;
	float* m_activation_Buffer;
	float* m_result_Buffer;
	NeuralNetworkData* m_nnd_Buffer;
	NeuralSwapData* m_nld_Buffer;
	NeuralNetworkData m_nnd;
};

#endif //!__NEURAL_NETWORK__