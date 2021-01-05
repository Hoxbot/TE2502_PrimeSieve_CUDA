#include "sieve_sundaram_cuda.cuh"

//#include "../support/cuda_error_output.h"

//CUDA---------------------------------------------------------------------------------------------
__global__ void SundaramKernel(size_t in_start, size_t in_n, bool* in_device_memory) {
	//Get the thread's index
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;

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
void SieveSundaramCUDA::SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr) {
	SundaramKernel <<<in_blocks, in_threads, 0>>> (in_start, in_end, in_mem_ptr);
}

void SieveSundaramCUDA::DoSieve() {

	//Allocate
	this->AllocateGPUMemory();

	this->timer_.SaveTime();

	//Upload
	this->UploadMemory();

	this->timer_.SaveTime();

	//Launch work-groups
	this->LaunchKernel(this->start_);

	this->timer_.SaveTime();

	//Download
	this->DownloadMemory();

	this->timer_.SaveTime();

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

	this->mem_class_ptr_ = new PrimeMemoryBool(mem_size);
	//this->mem_class_ptr_ = new PrimeMemoryBit(mem_size);
	this->LinkMemory(this->mem_class_ptr_);

	//Sundaram starts all as primes
	this->mem_class_ptr_->SetAllPrime();

	this->timer_.SaveTime();

	this->DoSieve();

	this->timer_.SaveTime();
}

SieveSundaramCUDA::SieveSundaramCUDA(size_t in_n, PrimeMemoryFragsafe * in_ptr)// {
	: SieveBase(1, in_n), SieveCUDA() {

	//Determine memory capacity needed
	//NTS: +1 since we round up
	size_t mem_size = ((in_n - 2) / 2) + ((in_n - 2) % 2);

	//Set fragsafe memory
	in_ptr->AllocateSubMemory(mem_size);
	this->mem_class_ptr_ = in_ptr;
	this->LinkMemory(this->mem_class_ptr_);

	//Sundaram starts all as primes
	this->mem_class_ptr_->SetAllPrime();

	this->timer_.SaveTime();

	this->DoSieve();

	this->timer_.SaveTime();
}

SieveSundaramCUDA::~SieveSundaramCUDA() {
	//Do not delete memory if its a fragsafe pointer
	if (dynamic_cast<PrimeMemoryFragsafe*>(this->mem_class_ptr_) != nullptr) { return; }

	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}	
}

bool SieveSundaramCUDA::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	
	//Anything smaller than 2 is not a prime
	if (in_num < 2) { return false; }

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
