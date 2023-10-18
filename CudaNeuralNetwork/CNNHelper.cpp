#include "CNNHelper.hpp"

void CNNHelper::KernelDispath(int size, int deviceLimitBlockSize, int* numBlocks, int* blockSize)
{
    *numBlocks = (size + deviceLimitBlockSize - 1) / deviceLimitBlockSize;
    *blockSize = std::min(deviceLimitBlockSize, size);
}