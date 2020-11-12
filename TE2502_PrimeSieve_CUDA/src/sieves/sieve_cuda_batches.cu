#include "sieve_cuda_batches.cuh"

#include "../support/cuda_error_output.h"

//Private------------------------------------------------------------------------------------------

//Protected----------------------------------------------------------------------------------------
void SieveCUDABatches::AllocateGPUMemory() {
	//Get GPU limitations
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t gpu_global_mem_capacity = prop.totalGlobalMem;

	//Fetch the number of bytes stored on the CPU side memory
	size_t bytes_to_allocate = this->sieve_mem_ptr_->BytesAllocated();

	//If the more bytes are required than the GPU can hold
	//we index additional batches, partitioning the numbers
	size_t batch_num = 0;
	void* mem_ptr = this->sieve_mem_ptr_->getMemPtr();
	while (bytes_to_allocate > gpu_global_mem_capacity) {

		Batch b;

		size_t offset = batch_num * gpu_global_mem_capacity;

		b.batch_ptr = static_cast<bool*>(mem_ptr) + offset;	//Batch starts a number of bytes into cpu memory
		b.batch_size = gpu_global_mem_capacity;				//Size of batch
		b.batch_start_index = offset;						//The index (in cpu memory) the memory starts at

		this->batches_.push_back(b);

		batch_num++;
		bytes_to_allocate -= gpu_global_mem_capacity;
	}

	//Repeat process for the one batch that isn't overfull 
	Batch b;
	size_t offset = batch_num * gpu_global_mem_capacity;
	b.batch_ptr = static_cast<bool*>(mem_ptr) + offset;		//Batch starts a number of bytes into cpu memory
	b.batch_size = bytes_to_allocate;						//Size of batch
	b.batch_start_index = offset;							//The index (in cpu memory) the memory starts at
	this->batches_.push_back(b);
	batch_num++;

	//Allocate memory on device
	CUDAErrorOutput(
		cudaMalloc(
		(void**)&(this->device_mem_ptr_),
			this->batches_[0].batch_size	//The first batch is always large enough. Either it is maxed out and
		),									//all other batches are the same or smaller, or it is the only batch.
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

void SieveCUDABatches::UploadMemory() {
	//Upload batch on given index
	CUDAErrorOutput(
		cudaMemcpy(
			this->device_mem_ptr_,					//Target
			this->batches_[in_i].batch_ptr,			//Source
			this->batches_[in_i].batch_size,		//Byte count
			cudaMemcpyHostToDevice					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);
}

void SieveCUDABatches::DownloadMemory() {
	//Download batch on given index
	CUDAErrorOutput(
		cudaMemcpy(
			this->batches_[in_i].batch_ptr,			//Target
			this->device_mem_ptr_,					//Source
			this->batches_[in_i].batch_size,		//Byte count
			cudaMemcpyHostToDevice					//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);
}

void SieveCUDABatches::LaunchKernel(size_t in_sieve_start) {
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
	size_t alt_start = in_sieve_start;

	//Launch full blocks with 1024 threads	
	//NTS: A kernel can have 48 blocks at maximum? : no : 2^31 - 1?
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
SieveCUDABatches::SieveCUDABatches() {
}

SieveCUDABatches::~SieveCUDABatches() {
	//NTS: Do not delete this ptr here
	this->sieve_mem_ptr_ = nullptr;
}

void SieveCUDABatches::LinkMemory(PrimeMemoryBool * in_ptr) {
	this->sieve_mem_ptr_ = in_ptr;
}
