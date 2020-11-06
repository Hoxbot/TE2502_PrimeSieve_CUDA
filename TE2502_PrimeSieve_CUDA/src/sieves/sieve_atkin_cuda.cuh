#ifndef SIEVE_ATKIN_CUDA_H
#define SIEVE_ATKIN_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sieve_base.h"

//CUDA Stuff
__global__ void AtkinKernel(void* in_device_memory);

//Class
class SieveAtkinCUDA : public SieveBase {
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
	SieveAtkinCUDA(size_t in_n);
	~SieveAtkinCUDA();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_ATKIN_CUDA_H


