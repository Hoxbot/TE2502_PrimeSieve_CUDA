#ifndef SIEVE_SUNDARAM_CUDA_H
#define SIEVE_SUNDARAM_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sieve_base.h"
#include "sieve_cuda.cuh"

//CUDA Stuff
__global__ void SundaramKernel(size_t in_start, size_t in_n, bool* in_device_memory);

//Class
class SieveSundaramCUDA : public SieveBase, public SieveCUDA {
private:
	//size_t start_ = 0;
	//size_t end_ = 0;

	/* NTS: Moved GPU allocation to own class
	
	bool* device_mem_ptr_ = nullptr;

	void AllocateGPUMemory();
	void DeallocateGPUMemory();
	void UploadMemory();
	void DownloadMemory();
	void LaunchKernel();
	*/

	void SieveKernel(size_t in_blocks, size_t in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr);

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveSundaramCUDA(size_t in_n);
	~SieveSundaramCUDA();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_SUNDARAM_CUDA_H