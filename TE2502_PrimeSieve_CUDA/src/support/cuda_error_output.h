#ifndef CUDA_ERROR_OUTPUT_H
#define CUDA_ERROR_OUTPUT_H

#include <iostream>
#include <string>

#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

inline void CUDAErrorOutput(cudaError_t in_err, std::string in_msg, std::string in_func) {
	if (in_err != cudaSuccess) {
		std::cerr << ("CUDA Error: " + in_msg + " in " + in_func + "\n");
	}
}

#endif // !CUDA_ERROR_OUTPUT_H



