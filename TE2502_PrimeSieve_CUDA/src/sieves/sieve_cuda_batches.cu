#include "sieve_cuda_batches.cuh"

#include <algorithm>

#include "../support/cuda_error_output.h"

//Private------------------------------------------------------------------------------------------

//Protected----------------------------------------------------------------------------------------
void SieveCUDABatches::AllocateGPUMemory(size_t in_sieve_start, size_t in_sieve_end) {
	//Get GPU limitations
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t gpu_global_mem_capacity = prop.totalGlobalMem;

	//
	//NTS:	In current experiments we already only use 1/3rd
	//		By dividing on 9 we get a third of that
	//gpu_global_mem_capacity /= 9;
	//gpu_global_mem_capacity = 1025;
	gpu_global_mem_capacity = 1000;
	//	  100000000 <- Breaks when sending in this as n, with any small memory maximum

	//Fetch the number of bytes stored on the CPU side memory
	size_t bytes_to_allocate = this->sieve_mem_ptr_->BytesAllocated();

	//Either one batch with x threads id enough, or we need y batches with max threads
	this->threads_per_batch_ = std::min(bytes_to_allocate, gpu_global_mem_capacity);

	size_t batches_required = (bytes_to_allocate / gpu_global_mem_capacity);		//Number of full batches
	if (bytes_to_allocate % gpu_global_mem_capacity != 0) { batches_required++; }	//If there are any leftover numbers

	//Index the batches to be launched with:
	//>	A pointer to the memory they start at
	//>	The number they start sieving at
	//>	The number they stop sieving at
	//size_t batch_num = 0;
	bool* mem_ptr = static_cast<bool*>(this->sieve_mem_ptr_->getMemPtr());

	

	for (size_t batch_num = 0; batch_num < batches_required; batch_num++) {
		//Create batch and calculate its offset
		Batch b;
		size_t offset = batch_num * threads_per_batch_;

		//Calculate the adress in the memory the batch starts at
		b.batch_ptr = mem_ptr + offset;

		//Calculate the first and last number that is part of the batch
		b.batch_start_number = in_sieve_start + offset;
		b.batch_end_number = std::min(b.batch_start_number + threads_per_batch_ - 1, in_sieve_end);

		//Save batch
		this->batches_.push_back(b);
	}


	//Allocate memory on device
	CUDAErrorOutput(
		cudaMalloc(
		(void**)&(this->device_mem_ptr_),
			threads_per_batch_
		),
		"cudaMalloc()", __FUNCTION__
	);
}


void SieveCUDABatches::DeallocateGPUMemory() {
	//Deallocate the memory on device
	CUDAErrorOutput(
		cudaFree(this->device_mem_ptr_),
		"cudaFree()", __FUNCTION__
	);
	this->device_mem_ptr_ = nullptr;
}

void SieveCUDABatches::UploadMemory(size_t in_i) {
	
	//Calc batch size
	size_t batch_size = this->batches_[in_i].batch_end_number - this->batches_[in_i].batch_start_number + 1;

	//Upload batch on given index
	CUDAErrorOutput(
		cudaMemcpy(
			this->device_mem_ptr_,					//Target
			this->batches_[in_i].batch_ptr,			//Source
			batch_size,								//Byte count
			cudaMemcpyHostToDevice					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);
}

void SieveCUDABatches::DownloadMemory(size_t in_i) {
	
	//Calc batch size
	size_t batch_size = this->batches_[in_i].batch_end_number - this->batches_[in_i].batch_start_number + 1;

	//Download batch on given index
	CUDAErrorOutput(
		cudaMemcpy(
			this->batches_[in_i].batch_ptr,			//Target
			this->device_mem_ptr_,					//Source
			batch_size,								//Byte count
			cudaMemcpyDeviceToHost					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);
}

void SieveCUDABatches::LaunchKernel(size_t in_batch_index) {
	// Launch a kernel on the GPU with one thread for each element.
	//	->	block
	//	->	threads per block (max 1024)
	//	->	size of shared memory
		//NTS:	unsigned int, not size_t. Need to fix safe conversion?
		//		Excess threads are fine, cannot be more than 1024 which fits
	unsigned int full_blocks = this->threads_per_batch_ / 1024;		//Number of full blocks in this batch
	unsigned int excess_threads = this->threads_per_batch_ % 1024;	//Number of threads not handled by full blocks

	//The number where sieving should start at in this batch
	size_t alt_start = this->batches_[in_batch_index].batch_start_number;

	//The number where sieving should end in this batch
	size_t alt_end = this->batches_[in_batch_index].batch_end_number;

	//Launch full blocks with 1024 threads
	unsigned int max_blocks = 2147483647;
	while (full_blocks > 0) {

		//Determine number of blocks in launch
		unsigned int blocks_in_launch = (full_blocks > max_blocks) ? max_blocks : full_blocks;

		//Launch kernel
		std::cout << ">>\tLaunching [" << blocks_in_launch << " of " << full_blocks << "] full blocks\n";
		this->SieveKernel(blocks_in_launch, 1024, alt_start, alt_end, in_batch_index, this->device_mem_ptr_);

		//Decrease number of remaining blocks
		//Move kernel starting value
		full_blocks -= blocks_in_launch;
		alt_start += blocks_in_launch * 1024;

		// Check for any errors launching the kernel
		CUDAErrorOutput(
			cudaGetLastError(),
			"<full blocks launch>",
			__FUNCTION__
		);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		CUDAErrorOutput(
			cudaDeviceSynchronize(),
			"cudaDeviceSynchronize()",
			__FUNCTION__
		);

	}

	//Launch leftover threads in 1 block //NTS: Will run sequentially, thus start and end must be altered
	if (excess_threads > 0) {
		std::cout << ">>\tLaunching [" << excess_threads << "] excess threads\n";
		this->SieveKernel(1, excess_threads, alt_start, alt_end, in_batch_index, this->device_mem_ptr_);

		// Check for any errors launching the kernel
		CUDAErrorOutput(
			cudaGetLastError(),
			"<excess thread launch>",
			__FUNCTION__
		);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		CUDAErrorOutput(
			cudaDeviceSynchronize(),
			"cudaDeviceSynchronize()",
			__FUNCTION__
		);
	}


}

//Public-------------------------------------------------------------------------------------------
SieveCUDABatches::SieveCUDABatches() {
}

SieveCUDABatches::~SieveCUDABatches() {
	//NTS: Do not delete this ptr here
	this->sieve_mem_ptr_ = nullptr;
}

void SieveCUDABatches::LinkMemory(PrimeMemoryBool * in_ptr) {
	this->sieve_mem_ptr_ = in_ptr;
}
