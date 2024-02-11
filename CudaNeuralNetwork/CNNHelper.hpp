//Peer Programming: Guo, Albarello
#ifndef __CNN_HELPER__
#define __CNN_HELPER__

#include <iostream>
#include <cuda_runtime.h>

class CNNHelper
{
public:
	static void KernelDispath(int size, cudaDeviceProp* deviceProp, dim3* numBlocks, dim3* blockSize);
	static void KernelDispathDim3(dim3 size, cudaDeviceProp* deviceProp, dim3* numBlocks, dim3* blockSize);
};

#endif