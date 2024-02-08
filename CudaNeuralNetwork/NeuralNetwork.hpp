#ifndef __NEURAL_NETWORK__
#define __NEURAL_NETWORK__

#include <iostream>
#include <vector>
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"
#include <map>

class NeuralNetwork
{
public:
	NeuralNetwork(NeuralNetworkData nnd);
	~NeuralNetwork();
	void trainingDataSet(const std::map<const std::string, std::vector<float*>>& data, int input_size,float min_percent_error_train);
	void trainingInput(const std::vector<std::vector<float>> input, const std::vector<std::vector<float>> output, std::vector<float>* error, float min_percent_error_train);
	void useInput(const std::vector<std::vector<float>> input, std::vector<std::vector<float>>* output);
	void useInputImage(float* col, std::vector<float>* output);
	void loadModel(const std::string& modelPath);
	void saveModel(const std::string& modelPath);
	void propagate();
	void backPropagate(std::vector<float> prediction_Data);
	NeuralNetworkData* getNeuralNetworkData();
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