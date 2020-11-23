#include "sieve_sundaram_cuda_batches.cuh"

//CUDA---------------------------------------------------------------------------------------------
__global__ void SundaramBatchKernel(
	size_t in_start, 
	size_t in_end, 
	size_t in_generation, 
	size_t in_batch_size, 
	bool* in_device_memory
) {
	//Get the thread's index
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	//Test: Flips every other generation true/false
	//in_device_memory[i] = (in_generation % 2 == 0);

	//---BATCH CURRENT GENERATION---
	//The first cuda thread has id 0
	//We offset by in_start (in the very beginning this is 1 since Sundaram starts at 1)
	i += in_start;

	//De-list all numbers that fullful the condition: (i + j + 2*i*j) <= n
	for (size_t j = i; (i + j + 2 * i*j) <= in_end; j++) {
		in_device_memory[(i + j + 2 * i*j) - in_start] = false;		// NTS: (-in_start) offsets to correct array index
	}

	//---BATCH EARLIER GENERATIONS---
	//>	Earlier batches will not have had access to the memory space of this batch.
	//>	For each earlier batch with the same thread position (ergo: i), find the
	//	first j that reaches into this batch's memory space
	//>	Iterate j:s until we reach the end of the batch
	for (size_t g = 0; g < in_generation; g++) {
		//Jump back one batch size to find the i of the previous generation
		i =- in_batch_size;

		//Compute which j is the first to reach into the current batch's memory space
		//j >= i, so we never start from a j less than i (thus fmaxf())
		float j_start = fmaxf(ceilf((in_start - i) / (2 * i + 1)), i);

		//Run iterations until we reach the end of span (in_end)
		for (size_t j = j_start; (i + j + 2*i*j) <= in_end; j++) {
			in_device_memory[(i + j + 2*i*j) - in_start] = false;		// NTS: (-in_start) offsets to correct array index
		}
	}
}


//Private------------------------------------------------------------------------------------------
void SieveSundaramCUDABatches::SieveKernel(
	unsigned int in_blocks, 
	unsigned int in_threads, 
	size_t in_start, 
	size_t in_end, 
	size_t in_generation, 
	bool * in_mem_ptr
) {
	SundaramBatchKernel <<<in_blocks, in_threads, 0>>> (in_start, in_end, in_generation, this->threads_per_batch_, in_mem_ptr);
}

void SieveSundaramCUDABatches::DoSieve() {
	//Allocate
	this->AllocateGPUMemory(this->start_, this->end_);

	for (size_t i = 0; i < this->batches_.size(); i++) {
		//Upload batch
		this->UploadMemory(i);

		//Launch work-groups
		this->LaunchKernel(i);

		//Download batch
		this->DownloadMemory(i);
	}

	//Deallocate
	this->DeallocateGPUMemory();

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
