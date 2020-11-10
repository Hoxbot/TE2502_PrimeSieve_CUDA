#ifndef SIEVE_CUDA_H
#define SIEVE_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "prime_memory/prime_memory_bool.h"

//Class
class SieveCUDA {
private:
	PrimeMemoryBool* sieve_mem_ptr_ = nullptr;
	bool* device_mem_ptr_ = nullptr;

protected:
	void AllocateGPUMemory();
	void DeallocateGPUMemory();
	void UploadMemory();
	void DownloadMemory();
	void LaunchKernel(size_t in_sieve_start);

	virtual void SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr) = 0;

public:
	SieveCUDA();
	~SieveCUDA();

	void LinkMemory(PrimeMemoryBool* in_ptr);
};

#endif // !SIEVE_CUDA_H