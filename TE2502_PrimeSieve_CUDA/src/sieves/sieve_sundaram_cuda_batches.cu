#include "sieve_sundaram_cuda_batches.cuh"

//CUDA---------------------------------------------------------------------------------------------
__global__ void SundaramKernel(size_t in_start, size_t in_n, bool* in_device_memory, size_t in_batch_offset) {

}


//Private------------------------------------------------------------------------------------------
void SieveSundaramCUDABatches::SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool * in_mem_ptr) {
	//SundaramKernel << <in_blocks, in_threads, 0 >> > (in_start, in_end, in_mem_ptr);
}

void SieveSundaramCUDABatches::DoSieve() {
	//WORKING HERE: Fix batch scheduling and rework Sundaram CUDA function
}

size_t SieveSundaramCUDABatches::IndexToNumber(size_t in_i) {
	return 2 * (in_i + this->start_) + 1;
}

//Public-------------------------------------------------------------------------------------------
SieveSundaramCUDABatches::SieveSundaramCUDABatches(size_t in_n)// {
	: SieveBase(1, in_n), SieveCUDABatches() {

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

SieveSundaramCUDABatches::~SieveSundaramCUDABatches() {
	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}
}

bool SieveSundaramCUDABatches::IsPrime(size_t in_num) {
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
