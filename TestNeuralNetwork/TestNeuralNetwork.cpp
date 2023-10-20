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

    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };
    AddWithCudaFunc addWithCuda = (AddWithCudaFunc)GetProcAddress(hDll, "addWithCuda");
    CreateNeuralNetwork createNeuralNetwork = (CreateNeuralNetwork)GetProcAddress(hDll, "createNeuralNetwork");
    NeuralNetworkData nnd{};
    nnd.nb_input_layer = 2;
    nnd.nb_col_hiden_layer = 4;
    nnd.nb_hiden_layer = 4;
    nnd.nb_output_layer = 1;
    NeuralNetwork * nn = createNeuralNetwork(nnd);    
    addWithCuda(c, a, b, arraySize);
    for (int i = 0; i < arraySize;i++)
    {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    if (addWithCuda == NULL) 
    {
        return 1;
    }
    return 0;
}