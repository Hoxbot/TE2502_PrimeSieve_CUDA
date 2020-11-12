#ifndef SIEVE_SUNDARAM_CUDA_BATCHES_H
#define SIEVE_SUNDARAM_CUDA_BATCHES_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sieve_base.h"
#include "sieve_cuda_batches.cuh"

//CUDA Stuff
__global__ void SundaramKernel(size_t in_start, size_t in_n, bool* in_device_memory, size_t in_batch_offset);

//Class
class SieveSundaramCUDABatches : public SieveBase, public SieveCUDABatches {
private:
	void SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr);

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveSundaramCUDABatches(size_t in_n);
	~SieveSundaramCUDABatches();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_SUNDARAM_CUDA_BATCHES_H
