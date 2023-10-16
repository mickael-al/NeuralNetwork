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
    addWithCuda(c, a, b, arraySize);
    
    if (addWithCuda == NULL) 
    {
        return 1;
    }
    return 0;
}