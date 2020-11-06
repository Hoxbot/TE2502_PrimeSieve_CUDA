#include "sieve_atkin_cuda.cuh"

//#include "../support/cuda_error_output.h"

//CUDA---------------------------------------------------------------------------------------------
__global__ void AtkinKernel(void* in_device_memory) {

}

//Private------------------------------------------------------------------------------------------
/*

void SieveAtkinCUDA::AllocateGPUMemory() {
	//Allocate memory on device
	CUDAErrorOutput(
		cudaMalloc(
		(void**)&(this->device_mem_ptr_),
			this->mem_class_ptr_->BytesAllocated()
		),
		"cudaMalloc()", __FUNCTION__
	);
}

void SieveAtkinCUDA::DeallocateGPUMemory() {
	//Deallocate the memory on device
	CUDAErrorOutput(
		cudaFree(this->device_mem_ptr_),
		"cudaFree()", __FUNCTION__
	);
	this->device_mem_ptr_ = nullptr;
}

void SieveAtkinCUDA::UploadMemory() {
	//Copy data to memory
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

void SieveAtkinCUDA::DownloadMemory() {
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

void SieveAtkinCUDA::LaunchKernel() {
}

*/

void SieveAtkinCUDA::DoSieve() {
}

size_t SieveAtkinCUDA::IndexToNumber(size_t in_i) {
	return this->start_ + in_i;
}

//Public-------------------------------------------------------------------------------------------
SieveAtkinCUDA::SieveAtkinCUDA(size_t in_n) 
	: SieveBase(1, in_n) {

	//NTS: Atkins excluding limit? ( [1, n[ )

	//Determine memory capacity needed
	size_t mem_size = in_n;

	this->mem_class_ptr_ = new PrimeMemoryBool(mem_size);
	//this->mem_class_ptr_ = new PrimeMemoryBit(mem_size);

	//Atkin starts all as non-primes
	this->mem_class_ptr_->SetAllNonPrime();

	this->private_timer_.SaveTime();

	//Set 2 and 3 manually as sieving process starts at 5
	if (in_n >= 2) { this->mem_class_ptr_->SetPrime(1); }
	if (in_n >= 3) { this->mem_class_ptr_->SetPrime(2); }

	this->DoSieve();

	this->private_timer_.SaveTime();

}

SieveAtkinCUDA::~SieveAtkinCUDA() {
	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}
}

bool SieveAtkinCUDA::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	//Otherwise return the stored bool for that value

	//Offset number to correct index
	size_t the_number_index = in_num - this->start_;

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}
