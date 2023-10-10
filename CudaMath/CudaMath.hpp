#pragma once

#ifndef EXP_CUDA_MATH

	#ifndef BUILD_CUDA_MATH

		#pragma comment(lib, "CudaMath.lib")
		#define EXP_CUDA_MATH __declspec(dllimport)

	#else

		#define EXP_CUDA_MATH __declspec(dllexport)

	#endif

#endif

extern "C" EXP_CUDA_MATH bool add(int* c, const int* a, const int* b, int arraySize);