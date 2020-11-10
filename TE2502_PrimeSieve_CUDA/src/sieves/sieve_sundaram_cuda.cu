#include "sieve_sundaram_cuda.cuh"

//#include "../support/cuda_error_output.h"

//CUDA---------------------------------------------------------------------------------------------
__global__ void SundaramKernel(size_t in_start, size_t in_n, bool* in_device_memory) {
	//Get the thread's index
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	//The first cuda thread has id 0
	//We offset by in_start (in the very beginning this is 1 since Sundaram starts at 1)
	i += in_start;

	//De-list all numbers that fullful the condition: (i + j + 2*i*j) <= n
	for (size_t j = i; (i + j + 2*i*j) <= in_n; j++) {
		in_device_memory[(i + j + 2 * i*j) - 1] = false;		// NTS: (-1) offsets to correct array index since indexing starts at 0
	}

	//Wait for all kernels to update
	//__syncthreads();

}

//Private------------------------------------------------------------------------------------------
/*

void SieveSundaramCUDA::AllocateGPUMemory() {
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
			this->mem_class_ptr_->BytesAllocated()
		),
		"cudaMalloc()", __FUNCTION__
	);
}

void SieveSundaramCUDA::DeallocateGPUMemory() {
	//Deallocate the memory on device
	CUDAErrorOutput(
		cudaFree(this->device_mem_ptr_),
		"cudaFree()", __FUNCTION__
	);
	this->device_mem_ptr_ = nullptr;
}

void SieveSundaramCUDA::UploadMemory() {
	//Copy data to memory
	//NTS: The booleans must be true, but is an upload needed?
	//Is it enough to allocate, alter in the kernel and download?
	CUDAErrorOutput(
		cudaMemcpy(
			this->device_mem_ptr_,					//Target
			this->mem_class_ptr_->getMemPtr(),		//Source
			this->mem_class_ptr_->BytesAllocated(),	//Byte count
			cudaMemcpyHostToDevice					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);
}

void SieveSundaramCUDA::DownloadMemory() {

	//NTS: Might need to be altered to dowload to a specific
	// array index and forward

	//Download data into memory structure
	CUDAErrorOutput(
		cudaMemcpy(
			this->mem_class_ptr_->getMemPtr(),		//Target
			this->device_mem_ptr_,					//Source
			this->mem_class_ptr_->BytesAllocated(),	//Byte count
			cudaMemcpyDeviceToHost					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);

}

void SieveSundaramCUDA::LaunchKernel() {
	// Launch a kernel on the GPU with one thread for each element.
	//	->	block
	//	->	threads per block (max 1024)
	//	->	size of shared memory
	size_t full_blocks = this->mem_class_ptr_->NumberCapacity() / 1024;			//Number of full blocks
	size_t excess_threads = this->mem_class_ptr_->NumberCapacity() % 1024;		//Number of threads not handled by full blocks
	//size_t bytes = this->mem_class_ptr_->BytesAllocated();					//Number of bytes to be in shared memory //NTS: Everything is in global, no shared needed 

	//Get where sieving should end
	size_t n = this->mem_class_ptr_->NumberCapacity();

	//If there are to be several kernel launches we need to figure out
	//where the subsequent blocks should start
	size_t alt_start = this->start_;

	//Launch full blocks with 1024 threads	//NTS: A kernel can have 48 blocks at maximum? : no : 2^31 - 1?
	size_t max_blocks = 2147483647;
	//size_t max_blocks = 10;
	while (full_blocks > 0) {

		//Determine number of blocks in launch
		size_t blocks_in_launch = (full_blocks > max_blocks) ? max_blocks : full_blocks;

		//Launch kernel
		std::cout << ">>\tLaunching [" << blocks_in_launch << " of " << full_blocks << "] full blocks\n";
		SundaramKernel <<<blocks_in_launch, 1024, 0>>> (alt_start, n, this->device_mem_ptr_);

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
		SundaramKernel <<<1, excess_threads, 0>>>(alt_start, n, this->device_mem_ptr_);

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

*/

void SieveSundaramCUDA::SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr) {
	SundaramKernel <<<in_blocks, in_threads, 0>>> (in_start, in_end, in_mem_ptr);
}

void SieveSundaramCUDA::DoSieve() {

	//Allocate
	this->AllocateGPUMemory();

	this->private_timer_.SaveTime();

	//Upload
	this->UploadMemory();

	this->private_timer_.SaveTime();

	//Launch work-groups
	this->LaunchKernel(this->start_);

	this->private_timer_.SaveTime();

	//Download
	this->DownloadMemory();

	this->private_timer_.SaveTime();

	//Deallocate
	this->DeallocateGPUMemory();

}

size_t SieveSundaramCUDA::IndexToNumber(size_t in_i) {
	return 2*(in_i + this->start_) + 1;
}

//Public-------------------------------------------------------------------------------------------
SieveSundaramCUDA::SieveSundaramCUDA(size_t in_n)// {
	: SieveBase(1, in_n), SieveCUDA() {
	

	//Determine memory capacity needed
	//NTS: +1 since we round up
	size_t mem_size = ((in_n - 2) / 2) + ((in_n - 2) % 2);

	//WORKING HERE: check sieve, check if we need to start from 0/1 in sundaram

	this->mem_class_ptr_ = new PrimeMemoryBool(mem_size);
	//this->mem_class_ptr_ = new PrimeMemoryBit(mem_size);
	this->LinkMemory(this->mem_class_ptr_);

	//Sundaram starts all as primes
	this->mem_class_ptr_->SetAllPrime();

	this->private_timer_.SaveTime();

	this->DoSieve();

	this->private_timer_.SaveTime();
}

SieveSundaramCUDA::~SieveSundaramCUDA() {
	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}	
}

bool SieveSundaramCUDA::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	
	//Sundaram's sieve does not store even numbers
	//> 2 special case
	//> All other even numbers false
	if (in_num == 2) { return true; }
	if ((in_num % 2) == 0) { return false; }

	//For odd numbers, offset number to correct index
	size_t the_number_index = ((in_num - 1) / 2) - this->start_;

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}
