//Peer Programming: Guo, Albarello
#ifndef __CUDA_NEURAL_NETWORK__
#define __CUDA_NEURAL_NETWORK__

#include "export.hpp"
#include "NeuralNetworkData.hpp"
#include "NeuralSwapData.hpp"
#include <string>
#include <vector>
#include "Vec.hpp"//et boom 

class NeuralNetwork;
class LinearModel;

LinearModel* createLinearModel();
typedef LinearModel* (*CreateLinearModel)();

void releaseLinearModel(LinearModel* lm);
typedef void(*ReleaseLinearModel)(LinearModel*);

void trainingLinearModel(LinearModel* lm, double learning_rate, Vec2* training_data, int size, std::vector<double> point, std::vector<float>* error);
typedef void(*TrainingLinearModel)(LinearModel*, double, Vec2*, int, std::vector<double>, std::vector<float>*);

double predictLinearModel(LinearModel* lm, Vec2* point);
typedef double(*PredictLinearModel)(LinearModel*,Vec2*);

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd);
typedef NeuralNetwork* (*CreateNeuralNetwork)(NeuralNetworkData nnd);

void trainingNeuralNetwork(NeuralNetwork* nn, const std::string& dataSetPath, std::vector<float>* error, double * min_percent_error_train);
typedef void (*TrainingNeuralNetwork)(NeuralNetwork*, const std::string&, std::vector<float>*, double*);

void trainingNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<double>> input, const std::vector<std::vector<double>> output, std::vector<float>*error, double* min_percent_error_train);
typedef void (*TrainingNeuralNetworkInput)(NeuralNetwork*, const std::vector<std::vector<double>>, const std::vector<std::vector<double>>, std::vector<float>* error, double*);

void useNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<double>> input,std::vector<std::vector<double>> * output);
typedef void (*UseNeuralNetworkInput)(NeuralNetwork*, const std::vector<std::vector<double>>, std::vector<std::vector<double>>* output);

void useNeuralNetworkImage(NeuralNetwork* nn, const std::string& image_path, std::vector<double>* output);
typedef void (*UseNeuralNetworkImage)(NeuralNetwork*, const std::string&, std::vector<double>*);

void loadNeuralNetworkModel(NeuralNetwork* nn, const std::string& modelPath);
typedef void(*LoadNeuralNetworkModel)(NeuralNetwork*, const std::string&);

void saveNeuralNetworkModel(NeuralNetwork* nn, const std::string& modelPath);
typedef void(*SaveNeuralNetworkModel)(NeuralNetwork*, const std::string&);

void releaseNeuralNetwork(NeuralNetwork* network);
typedef void(*ReleaseNeuralNetwork)(NeuralNetwork*);

void generateDataSet(const std::string& path, const std::string& dataSetSavepath,int image_data_size);
typedef void(*GenerateDataSet)(const std::string&, const std::string&, int);

void updateNNAlpha(NeuralNetwork* nn, double alpha);
typedef void(*UpdateNNAlpha)(NeuralNetwork*, double);

int addWithCuda(int* c, const int* a, const int* b, unsigned int size);
typedef int(*AddWithCudaFunc)(int*, const int*, const int*, unsigned int);

#endif //!__CUDA_NEURAL_NETWORK__