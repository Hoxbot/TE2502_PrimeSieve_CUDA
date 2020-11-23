#ifndef SIEVE_CUDA_BATCHES_H
#define SIEVE_CUDA_BATCHES_H

#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "prime_memory/prime_memory_bool.h"

//Class
class SieveCUDABatches {
private:
	struct Batch {
		//Batch(size_t i, size_t s) { batch_index = i; batch_size = s; }
		bool* batch_ptr;
		size_t batch_size;
		
		size_t batch_start_number;
		size_t batch_end_number;
	};

	//size_t gpu_global_mem_capacity_;
	//size_t gpu_block_capacity_; not available from any structure

	PrimeMemoryBool* sieve_mem_ptr_ = nullptr;
	bool* device_mem_ptr_ = nullptr;

protected:
	std::vector<Batch> batches_;
	size_t threads_per_batch_;

	void AllocateGPUMemory(size_t in_sieve_start, size_t in_sieve_end);
	void DeallocateGPUMemory();
	void UploadMemory(size_t in_i);
	void DownloadMemory(size_t in_i);
	void LaunchKernel(size_t in_batch_index);

	virtual void SieveKernel(
		unsigned int in_blocks,
		unsigned int in_threads,
		size_t in_start,
		size_t in_end,
		size_t in_generation,
		bool* in_mem_ptr
	) = 0;

public:
	SieveCUDABatches();
	~SieveCUDABatches();

	void LinkMemory(PrimeMemoryBool* in_ptr);
};

#endif // !SIEVE_CUDA_BATCHES_H