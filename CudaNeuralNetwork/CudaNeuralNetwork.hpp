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

void trainingLinearModel(LinearModel* lm, float learning_rate, Vec2* training_data, int size, std::vector<double> point, std::vector<float>* error);
typedef void(*TrainingLinearModel)(LinearModel*, float, Vec2*, int, std::vector<double>, std::vector<float>*);

double predictLinearModel(LinearModel* lm, Vec2* point);
typedef double(*PredictLinearModel)(LinearModel*,Vec2*);

NeuralNetwork* createNeuralNetwork(NeuralNetworkData nnd);
typedef NeuralNetwork* (*CreateNeuralNetwork)(NeuralNetworkData nnd);

void trainingNeuralNetwork(NeuralNetwork* nn, const std::string& dataSetPath, std::vector<float>* error, float * min_percent_error_train);
typedef void (*TrainingNeuralNetwork)(NeuralNetwork*, const std::string&, std::vector<float>*, float*);

void trainingNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<float>> input, const std::vector<std::vector<float>> output, std::vector<float>*error, float* min_percent_error_train);
typedef void (*TrainingNeuralNetworkInput)(NeuralNetwork*, const std::vector<std::vector<float>>, const std::vector<std::vector<float>>, std::vector<float>* error, float*);

void useNeuralNetworkInput(NeuralNetwork* nn, const std::vector<std::vector<float>> input,std::vector<std::vector<float>> * output);
typedef void (*UseNeuralNetworkInput)(NeuralNetwork*, const std::vector<std::vector<float>>, std::vector<std::vector<float>>* output);

void useNeuralNetworkImage(NeuralNetwork* nn, const std::string& image_path, std::vector<float>* output);
typedef void (*UseNeuralNetworkImage)(NeuralNetwork*, const std::string&, std::vector<float>*);

void loadNeuralNetworkModel(NeuralNetwork* nn, const std::string& modelPath);
typedef void(*LoadNeuralNetworkModel)(NeuralNetwork*, const std::string&);

void saveNeuralNetworkModel(NeuralNetwork* nn, const std::string& modelPath);
typedef void(*SaveNeuralNetworkModel)(NeuralNetwork*, const std::string&);

void releaseNeuralNetwork(NeuralNetwork* network);
typedef void(*ReleaseNeuralNetwork)(NeuralNetwork*);

void generateDataSet(const std::string& path, const std::string& dataSetSavepath,int image_data_size);
typedef void(*GenerateDataSet)(const std::string&, const std::string&, int);

void updateNNAlpha(NeuralNetwork* nn, float alpha);
typedef void(*UpdateNNAlpha)(NeuralNetwork*, float);

int addWithCuda(int* c, const int* a, const int* b, unsigned int size);
typedef int(*AddWithCudaFunc)(int*, const int*, const int*, unsigned int);

#endif //!__CUDA_NEURAL_NETWORK__