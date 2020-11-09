#include "sieve_cuda.cuh"

#include "../support/cuda_error_output.h"

//Private------------------------------------------------------------------------------------------

//Protected----------------------------------------------------------------------------------------
void SieveCUDA::AllocateGPUMemory() {
	//CUDA Memory Notes:
	// A single CUDA Block can run 1024 threads.
	// Each block shares:
	//	- The global memory (slow)
	//	- The constant memory (fast)
	// Each block has:
	//	- A shared memory that can be accessed by the threads in the block
	//	- A set of registers (for variables an the like i presume)
	//		> NTS: Keep in mind that since each thread can write into the registers
	//		> the numbers of variables declared in the kernel functions are multiplied
	//		> by the number of threads. Over-declaration of variables eats memory fast.
	//
	//	Global memory capacity (bytes): 3221225472
	//	Shared memory capacity (bytes): 49152


	//Step 0.1: 
	// NTS: Do we benefit from contious allocation?

	//Step 0.2:
	// NTS: For capacities over the GPU Global capacity
	//	> Use pointers into the array
	//	> Save neccesary numerical indexing?


	//Allocate memory on device
	CUDAErrorOutput(
		cudaMalloc(
		(void**)&(this->device_mem_ptr_),
			this->sieve_mem_ptr_->BytesAllocated()
		),
		"cudaMalloc()", __FUNCTION__
	);
}

void SieveCUDA::DeallocateGPUMemory() {
	//Deallocate the memory on device
	CUDAErrorOutput(
		cudaFree(this->device_mem_ptr_),
		"cudaFree()", __FUNCTION__
	);
	this->device_mem_ptr_ = nullptr;
}

void SieveCUDA::UploadMemory() {
	//Copy data to memory
	//NTS: The booleans must be true, but is an upload needed?
	//Is it enough to allocate, alter in the kernel and download?
	CUDAErrorOutput(
		cudaMemcpy(
			this->device_mem_ptr_,					//Target
			this->sieve_mem_ptr_->getMemPtr(),		//Source
			this->sieve_mem_ptr_->BytesAllocated(),	//Byte count
			cudaMemcpyHostToDevice					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);
}

void SieveCUDA::DownloadMemory() {

	//NTS: Might need to be altered to dowload to a specific
	// array index and forward

	//Download data into memory structure
	CUDAErrorOutput(
		cudaMemcpy(
			this->sieve_mem_ptr_->getMemPtr(),		//Target
			this->device_mem_ptr_,					//Source
			this->sieve_mem_ptr_->BytesAllocated(),	//Byte count
			cudaMemcpyDeviceToHost					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);

}

void SieveCUDA::LaunchKernel(size_t in_sieve_start) {
	// Launch a kernel on the GPU with one thread for each element.
	//	->	block
	//	->	threads per block (max 1024)
	//	->	size of shared memory
		//NTS:	unsigned int, not size_t. Need to fix safe conversion?
		//		Excess threads are fine, cannot be more than 1024 which fits
	unsigned int full_blocks = this->sieve_mem_ptr_->NumberCapacity() / 1024;	//Number of full blocks
	unsigned int excess_threads = this->sieve_mem_ptr_->NumberCapacity() % 1024;		//Number of threads not handled by full blocks
	//size_t bytes = this->mem_class_ptr_->BytesAllocated();					//Number of bytes to be in shared memory //NTS: Everything is in global, no shared needed 

	//Get where sieving should end
	size_t n = this->sieve_mem_ptr_->NumberCapacity();

	//If there are to be several kernel launches we need to figure out
	//where the subsequent blocks should start
	size_t alt_start = in_sieve_start;	// this->start_;

	//Launch full blocks with 1024 threads	//NTS: A kernel can have 48 blocks at maximum? : no : 2^31 - 1?
	unsigned int max_blocks = 2147483647;
	//size_t max_blocks = 2;
	while (full_blocks > 0) {

		//Determine number of blocks in launch
		unsigned int blocks_in_launch = (full_blocks > max_blocks) ? max_blocks : full_blocks;

		//Launch kernel
		std::cout << ">>\tLaunching [" << blocks_in_launch << " of " << full_blocks << "] full blocks\n";
		//SundaramKernel <<<blocks_in_launch, 1024, 0>>> (alt_start, n, this->device_mem_ptr_);
		this->SieveKernel(blocks_in_launch, 1024, alt_start, n, this->device_mem_ptr_);

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
		//SundaramKernel <<<1, excess_threads, 0>>> (alt_start, n, this->device_mem_ptr_);
		this->SieveKernel(1, excess_threads, alt_start, n, this->device_mem_ptr_);

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
SieveCUDA::SieveCUDA() {
}

SieveCUDA::~SieveCUDA() {
	//NTS: Do not delete this ptr here
	this->sieve_mem_ptr_ = nullptr;
}

void SieveCUDA::LinkMemory(PrimeMemoryBool * in_ptr) {
	this->sieve_mem_ptr_ = in_ptr;
}
