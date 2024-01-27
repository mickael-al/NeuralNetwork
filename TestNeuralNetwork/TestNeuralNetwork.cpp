#include <iostream>
#include <Windows.h>
#include "CudaNeuralNetwork.hpp"

int main()
{
    HMODULE hDll = LoadLibrary(L".\\CudaNeuralNetwork.dll");
    if (hDll == NULL) 
    {
        return 1;
    }
    
    CreateNeuralNetwork createNeuralNetwork = (CreateNeuralNetwork)GetProcAddress(hDll, "createNeuralNetwork");
    ReleaseNeuralNetwork releaseNeuralNetwork = (ReleaseNeuralNetwork)GetProcAddress(hDll, "releaseNeuralNetwork");
    TrainingNeuralNetworkInput trainingNeuralNetworkInput = (TrainingNeuralNetworkInput)GetProcAddress(hDll, "trainingNeuralNetworkInput");
    TrainingNeuralNetwork trainingNeuralNetwork = (TrainingNeuralNetwork)GetProcAddress(hDll, "trainingNeuralNetwork");
    GenerateDataSet generateDataSet = (GenerateDataSet)GetProcAddress(hDll, "generateDataSet");
    if (createNeuralNetwork == NULL)
    {
        std::cerr << "createNeuralNetwork not found" << std::endl;
        return 1;
    }
    if (releaseNeuralNetwork == NULL)
    {
        std::cerr << "releaseNeuralNetwork not found" << std::endl;
        return 1;
    }
    const std::string modelPath = "../DataSet/data.dataset";
    //generateDataSet("../DataSet", modelPath, 256);

    NeuralNetworkData nnd{};
    nnd.nb_input_layer = 256;
    nnd.nb_col_hiden_layer = 5;
    nnd.nb_hiden_layer = 50;
    nnd.nb_output_layer = 3;
    nnd.alpha = 0.01f;
    nnd.is_classification = false;
    NeuralNetwork* nn = createNeuralNetwork(nnd);
    trainingNeuralNetwork(nn, modelPath, 1.0f);
    releaseNeuralNetwork(nn);

    /*NeuralNetworkData nnd{};
    nnd.nb_input_layer = 2;
    nnd.nb_col_hiden_layer = 2;
    nnd.nb_hiden_layer = 10;
    nnd.nb_output_layer = 1;
    nnd.alpha = 0.01f;
    nnd.is_classification = false;
    NeuralNetwork * nn = createNeuralNetwork(nnd);    
    std::vector<std::vector<float>> xor_data;
    xor_data.push_back({ 0,0 });
    xor_data.push_back({ 1,0 });
    xor_data.push_back({ 0,1 });
    xor_data.push_back({ 1,1 });
    std::vector<std::vector<float>> xor_result_data;
    xor_result_data.push_back({ -1 });
    xor_result_data.push_back({ 1 });
    xor_result_data.push_back({ 1 });
    xor_result_data.push_back({ -1 });
    trainingNeuralNetworkInput(nn, xor_data, xor_result_data,1.0f);
    releaseNeuralNetwork(nn);*/
    return 0;
}