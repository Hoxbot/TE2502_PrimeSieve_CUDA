#include "sieve_eratosthenes_cuda.cuh"

//CUDA---------------------------------------------------------------------------------------------
__global__ void EratosthenesKernel(size_t in_start, size_t in_n, bool* in_device_memory) {
	//Get the thread's index
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;

	//The first cuda thread has id 0
	//We offset by in_start
	i += in_start;

	//NTS: Calculates for every i, even those that might already have been identified as composite
	for (size_t j = i * i; j <= in_n; j = j + i) {
		in_device_memory[j - in_start] = false;
	}
}

//Private------------------------------------------------------------------------------------------
void SieveEratosthenesCUDA::SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr) {
	EratosthenesKernel <<<in_blocks, in_threads, 0>>> (in_start, in_end, in_mem_ptr);

}

void SieveEratosthenesCUDA::DoSieve() {

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

size_t SieveEratosthenesCUDA::IndexToNumber(size_t in_i) {
	return this->start_ + in_i;
}

//Public-------------------------------------------------------------------------------------------
SieveEratosthenesCUDA::SieveEratosthenesCUDA(size_t in_n)// {
	: SieveBase(2, in_n), SieveCUDA() {

	//Determine memory capacity needed
	size_t mem_size = in_n - 2 + 1; //+1 because it's inclusive: [start, end]

	this->mem_class_ptr_ = new PrimeMemoryBool(mem_size);
	//this->mem_class_ptr_ = new PrimeMemoryBit(mem_size);
	this->LinkMemory(this->mem_class_ptr_);

	//Eratosthenes starts all as primes
	this->mem_class_ptr_->SetAllPrime();

	this->timer_.SaveTime();

	this->DoSieve();

	this->timer_.SaveTime();

}

SieveEratosthenesCUDA::~SieveEratosthenesCUDA() {
	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}
}

bool SieveEratosthenesCUDA::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	//Otherwise return the stored bool for that value

	//Offset number to correct index
	size_t the_number_index = in_num - this->start_;

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}

