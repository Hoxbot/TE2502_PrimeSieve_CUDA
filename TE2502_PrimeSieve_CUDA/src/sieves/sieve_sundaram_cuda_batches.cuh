#ifndef SIEVE_SUNDARAM_CUDA_BATCHES_H
#define SIEVE_SUNDARAM_CUDA_BATCHES_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sieve_base.h"
#include "sieve_cuda_batches.cuh"

#include "prime_memory/prime_memory_fragsafe.h"

//CUDA Stuff
__global__ void SundaramBatchKernel(
	size_t in_start, 
	size_t in_end, 
	size_t in_generation, 
	size_t in_batch_size, 
	bool* in_device_memory
);

//Class
class SieveSundaramCUDABatches : public SieveBase, public SieveCUDABatches {
private:
	void SieveKernel(
		unsigned int in_blocks,
		unsigned int in_threads,
		size_t in_start,
		size_t in_end,
		size_t in_generation,
		bool* in_mem_ptr
	);

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveSundaramCUDABatches(size_t in_n);
	SieveSundaramCUDABatches(size_t in_n, PrimeMemoryFragsafe* in_ptr);
	~SieveSundaramCUDABatches();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_SUNDARAM_CUDA_BATCHES_H
