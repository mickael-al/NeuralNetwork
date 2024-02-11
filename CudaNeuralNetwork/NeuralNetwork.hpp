//Peer Programming: Guo, Albarello
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
	void trainingDataSet(const std::map<const std::string, std::vector<double*>>& data, int input_size, std::vector<float>* error,double * min_percent_error_train);
	void trainingInput(const std::vector<std::vector<double>> input, const std::vector<std::vector<double>> output, std::vector<float>* error, double * min_percent_error_train);
	void useInput(const std::vector<std::vector<double>> input, std::vector<std::vector<double>>* output);
	void useInputImage(double* col, std::vector<double>* output);
	void loadModel(const std::string& modelPath);
	void saveModel(const std::string& modelPath);
	void propagate();
	void updateAlpha(double alpha);
	void backPropagate(std::vector<double> prediction_Data);
	NeuralNetworkData* getNeuralNetworkData();
private:
	//gpu
	double *** m_self_w;
	double ** m_self_x;
	double ** m_self_delta;
	int* m_self_d;
	NeuralSwapData* m_nld_Buffer;
	NeuralNetworkDataCompact* m_nndc_Buffer;
	//gpu

	NeuralNetworkData m_nnd;
	NeuralSwapData m_nld;
	NeuralNetworkDataCompact m_nndc;
	int* m_array_size_d;
	double* m_outDelta;
	double* m_activation;

};

#endif //!__NEURAL_NETWORK__