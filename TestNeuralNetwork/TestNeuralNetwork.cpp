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

    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    AddWithCudaFunc addWithCuda = (AddWithCudaFunc)GetProcAddress(hDll, "addWithCuda");
    if (addWithCuda == NULL)
    {
        std::cerr << "addWithCuda not found" << std::endl;
        return 1;
    }

    addWithCuda(c, a, b, arraySize);
    for (int i = 0; i < arraySize; i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;*/
    
    CreateNeuralNetwork createNeuralNetwork = (CreateNeuralNetwork)GetProcAddress(hDll, "createNeuralNetwork");
    ReleaseNeuralNetwork releaseNeuralNetwork = (ReleaseNeuralNetwork)GetProcAddress(hDll, "releaseNeuralNetwork");
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
    NeuralNetworkData nnd{};
    nnd.nb_input_layer = 2;
    nnd.nb_col_hiden_layer = 1;
    nnd.nb_hiden_layer = 2000;
    nnd.nb_output_layer = 1;
    nnd.alpha = 0.01f;
    nnd.is_classification = false;
    NeuralNetwork * nn = createNeuralNetwork(nnd);        
    const std::string modelPath = "test";
    //generateDataSet("C:/Users/micka/Documents/Projet/Cpp/NeuralNetwork/DataSet", "C:/Users/micka/Documents/Projet/Cpp/NeuralNetwork/DataSet/data.dataset");
    trainingNeuralNetwork(nn, modelPath);
    releaseNeuralNetwork(nn);
    return 0;
}