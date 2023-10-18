#ifndef __CNN_HELPER__
#define __CNN_HELPER__

#include <iostream>

class CNNHelper
{
public:
	static void KernelDispath(int size, int deviceLimitBlockSize, int* numBlocks, int* blockSize);
};

#endif