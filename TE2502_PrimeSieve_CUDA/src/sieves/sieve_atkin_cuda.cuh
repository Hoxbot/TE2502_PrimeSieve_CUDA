#ifndef SIEVE_ATKIN_CUDA_H
#define SIEVE_ATKIN_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sieve_base.h"
#include "sieve_cuda.cuh"

//CUDA Stuff
__global__ void AtkinKernel(size_t in_start, size_t in_n, bool* in_device_memory);

//Class
class SieveAtkinCUDA : public SieveBase, public SieveCUDA {
private:
	void SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr);

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveAtkinCUDA(size_t in_n);
	~SieveAtkinCUDA();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_ATKIN_CUDA_H


