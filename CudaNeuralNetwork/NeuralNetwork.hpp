#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <iostream>
#include <vector>
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"

class NeuralNetwork
{
public:
	NeuralNetwork(float* weight_buffer, float* activation_Buffer, NeuralNetworkData* nnd_Buffer, NeuralSwapData* nld_Buffer);
	~NeuralNetwork();
	void loadModel(const std::string& modelPath);
	void setInputData(const std::vector<double>& inputData);
	void saveModel(const std::string& modelPath);
	void propagate();
	void backPropagate();
private:
	float* m_weight_buffer;
	float* m_activation_Buffer;
	NeuralNetworkData* m_nnd_Buffer;
	NeuralSwapData* m_nld_Buffer;
};

#endif //!__NEURAL_NETWORK__