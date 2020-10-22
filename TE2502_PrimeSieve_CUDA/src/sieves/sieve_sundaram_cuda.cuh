#ifndef SIEVE_SUNDARAM_CUDA_H
#define SIEVE_SUNDARAM_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sieve_base.h"

//CUDA Stuff
//Helper
void CUDAErrorOutput(cudaError_t in_err, std::string in_msg, std::string in_func);
//Kernels
__global__ void SundaramKernel(size_t in_start, size_t in_end, void* in_device_memory);

//Class
class SieveSundaramCUDA : public SieveBase {
private:
	//size_t start_ = 0;
	//size_t end_ = 0;

	void* device_mem_ptr_ = nullptr;

	void AllocateGPUMemory();
	void DeallocateGPUMemory();
	void UploadMemory();
	void DownloadMemory();
	void LaunchKernel();

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveSundaramCUDA(size_t in_n);
	~SieveSundaramCUDA();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_SUNDARAM_CUDA_H