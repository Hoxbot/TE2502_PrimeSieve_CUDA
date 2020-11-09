#include "sieve_atkin_cuda.cuh"

//#include "../support/cuda_error_output.h"

//CUDA---------------------------------------------------------------------------------------------
__global__ void AtkinKernel(size_t in_start, size_t in_n, bool* in_device_memory) {
	//Get the thread's index
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	//The first cuda thread has id 0
	//-> It computes number i
	//-> It should set array index i-1
	i += in_start;	//NTS: Exchange i for x?

	//Sieve of Atkins
	//> For (x^2 < n) and (y^2 < n), x = 1,2,..., y = 1,2,...
	//> A number is prime if any of the following is true:
	//>> (z = 4*x*x + y*y) has odd number of solutions	AND	(z % 12 = 1) or (z % 12 = 5)
	//>> (z = 3*x*x + y*y) has odd number of solutions	AND	(z % 12 = 7)
	//>> (z = 3*x*x - y*y) has odd number of solutions	AND (x > y)	AND	(z % 12 = 11)
	//> Multiples of squares might have been marked, delist:
	//>> (z = x*x*y), x = 1,2,..., y = 1,2,...

	//NTS: An terrible amount of if-statements for a GPGPU kernel. 
	//- Path divergence will cause slowdown.
	//- Could one rewrite them as an assignment ?
	//- (ergo: 'if(x) set array = true' -> set array (x))
	//- Might overwrite already set correct result in some cases
	//- Since we only set 'true' maybe we can make it so it does not overwrite already true entries?
	
	if (i*i < in_n) {
		size_t x = i;
		for (size_t y = 1; y*y < in_n; y++) {

			size_t z = (4*x*x) + (y*y);
			if (z <= in_n && (z % 12 == 1 || z % 12 == 5)) { in_device_memory[z - 1] = true; }

			z = (3*x*x) + (y*y);
			if (z <= in_n && (z % 12 == 7)) { in_device_memory[z - 1] = true; }

			z = (3*x*x) - (y*y);
			if (z <= in_n && (x > y) && (z % 12 == 11)) { in_device_memory[z - 1] = true; }
		}
	}

	// NTS: Should this be in the GPGPU function?
	//Only for 5 and onwards. More path divergence :/
	if (i >= 5 && i*i < in_n) {
		size_t x = i;
		if (in_device_memory[x - 1]) {
			for (size_t y = x*x; y < in_n; y += x*x) {
				in_device_memory[y - 1] = false;
			}
		}
	}
}

//Private------------------------------------------------------------------------------------------
void SieveAtkinCUDA::SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr) {
	AtkinKernel <<<in_blocks, in_threads, 0>>> (in_start, in_end, in_mem_ptr);
}

void SieveAtkinCUDA::DoSieve() {
	//Allocate
	this->AllocateGPUMemory();

	//Upload
	this->UploadMemory();

	//Launch work-groups
	this->LaunchKernel(this->start_);

	//Download
	this->DownloadMemory();

	//Deallocate
	this->DeallocateGPUMemory();
}

size_t SieveAtkinCUDA::IndexToNumber(size_t in_i) {
	return this->start_ + in_i;
}

//Public-------------------------------------------------------------------------------------------
SieveAtkinCUDA::SieveAtkinCUDA(size_t in_n) 
	: SieveBase(1, in_n), SieveCUDA() {

	//NTS: Atkins excluding limit? ( [1, n[ )

	//Determine memory capacity needed
	size_t mem_size = in_n;

	this->mem_class_ptr_ = new PrimeMemoryBool(mem_size);
	//this->mem_class_ptr_ = new PrimeMemoryBit(mem_size);
	this->LinkMemory(this->mem_class_ptr_);

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
