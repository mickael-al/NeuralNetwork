#include <iostream>
#include <Windows.h>
#include "kernel.h"

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
    nnd.nb_col_hiden_layer = 4;
    nnd.nb_hiden_layer = 4;
    nnd.nb_output_layer = 2;
    NeuralNetwork * nn = createNeuralNetwork(nnd);    
    const std::string modelPath = "test";
    releaseNeuralNetwork(nn);
    return 0;
}