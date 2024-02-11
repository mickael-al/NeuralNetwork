//Peer Programming: Guo, Albarello
#include "CNNHelper.hpp"

void CNNHelper::KernelDispath(int size, cudaDeviceProp * deviceProp, dim3* numBlocks, dim3* blockSize)
{
    numBlocks->x = (size + deviceProp->maxThreadsPerBlock - 1) / deviceProp->maxThreadsPerBlock;
    numBlocks->y = 1;
    numBlocks->z = 1;
    blockSize->x = std::min(deviceProp->maxThreadsPerBlock, size);
    blockSize->y = 1;
    blockSize->z = 1;
}

void CNNHelper::KernelDispathDim3(dim3 size,cudaDeviceProp* deviceProp, dim3* numBlocks, dim3* blockSize)
{
  
}