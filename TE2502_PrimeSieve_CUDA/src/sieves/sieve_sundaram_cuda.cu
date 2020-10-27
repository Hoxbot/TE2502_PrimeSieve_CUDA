#include "sieve_sundaram_cuda.cuh"

#include <iostream>

//CUDA---------------------------------------------------------------------------------------------
void CUDAErrorOutput(cudaError_t in_err, std::string in_msg, std::string in_func) {
	if (in_err != cudaSuccess) {
		std::cerr << ("CUDA Error: " + in_msg + " in " + in_func + "\n");
	}
}

__global__ void SundaramKernel(size_t in_start, size_t in_end, void* in_device_memory) {

	//Cast to bool	//NTS: Preferably we do this on cpu side, and only once
	bool* mem_ptr = reinterpret_cast<bool*>(in_device_memory);

	//Get the thread's index
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	//De-list all numbers that fullful the condition: (i + j + 2*i*j) <= n
	for (size_t j = i; (i + j + 2*i*j) <= in_end; j++) {
		mem_ptr[(i + j + 2 * i*j)] = false;
	}

	//Wait for all kernels to update
	__syncthreads();

}

//Private------------------------------------------------------------------------------------------
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
	size_t blocks = 1;
	//	->	threads per block (max 1024)
	size_t threads = this->n_;
	//	->	size of shared memory
	size_t bytes = this->mem_class_ptr_->BytesAllocated();
	//Launch
	SundaramKernel<<<blocks, threads, bytes>>>(this->start_, this->end_, this->device_mem_ptr_);

	// Check for any errors launching the kernel
	CUDAErrorOutput(
		cudaGetLastError(),
		"cudaGetLastError()",
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

void SieveSundaramCUDA::DoSieve() {

	//Allocate
	this->AllocateGPUMemory();

	//Upload
	this->UploadMemory();

	//Launch work-groups
	this->LaunchKernel();

	//Download
	this->DownloadMemory();

	//Deallocate
	this->DeallocateGPUMemory();

}

size_t SieveSundaramCUDA::IndexToNumber(size_t in_i) {
	return 2*(this->start_ + in_i) + 1;
}

//Public-------------------------------------------------------------------------------------------
SieveSundaramCUDA::SieveSundaramCUDA(size_t in_n)// {
	: SieveBase(0, (((in_n-2)/2)+1) ) {
	//NTS: +1 since we round up

	this->mem_class_ptr_ = new PrimeMemoryBool(this->n_);
	//this->mem_class_ptr_ = new PrimeMemoryBit(this->n_);

	//Sundaram starts all as primes
	this->mem_class_ptr_->SetAllPrime();

	this->private_timer_.SaveTime();

	//this->DoSieve();

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
	//Otherwise return the stored bool for that value

	//Offset number to correct index
	size_t the_number_index = (in_num - 1) / 2; //NTS: study division here

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}
