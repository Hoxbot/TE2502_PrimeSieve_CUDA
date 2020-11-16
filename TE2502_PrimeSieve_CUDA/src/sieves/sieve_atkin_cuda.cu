#include "sieve_atkin_cuda.cuh"

//#include "../support/cuda_error_output.h"

//CUDA---------------------------------------------------------------------------------------------
__global__ void AtkinKernel(size_t in_start, size_t in_n, bool* in_device_memory) {
	//Get the thread's index
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	//The first cuda thread has id 0
	//-> It computes number x
	//-> It should set array index x-1
	x += in_start;

	//Sieve of Atkins
	//> For (x^2 <= n) and (y^2 <= n), x = 1,2,..., y = 1,2,...
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
	
	//NTS:	"Odd number of solutions", does that mean we should flip the state to the inverse?
	//		Two hits (even number of solutions) would then flip false->true->false
	//Ans:	Yes, apparently.

	if (x*x <= in_n) {
		for (size_t y = 1; y*y <= in_n; y++) {

			size_t z = (4*x*x) + (y*y);
			//if (z <= in_n && (z % 12 == 1 || z % 12 == 5)) { in_device_memory[z - 1] = true; }
			if (z <= in_n && (z % 12 == 1 || z % 12 == 5)) { in_device_memory[z - 1] = !in_device_memory[z - 1]; }

			z = (3*x*x) + (y*y);
			//if (z <= in_n && (z % 12 == 7)) { in_device_memory[z - 1] = true; }
			if (z <= in_n && (z % 12 == 7)) { in_device_memory[z - 1] = !in_device_memory[z - 1]; }

			z = (3*x*x) - (y*y);
			//if (z <= in_n && (x > y) && (z % 12 == 11)) { in_device_memory[z - 1] = true; }
			if (z <= in_n && (x > y) && (z % 12 == 11)) { in_device_memory[z - 1] = !in_device_memory[z - 1]; }
		}
	}

	// NTS: Should this be in the GPGPU function?
	// Only for 5 and onwards. More path divergence :/
	//
	// Other NTS: Error probably occurs here
	// Theory: These lines (that should correct to false) are run BEFORE the kernel
	// that incorrectly sets true. Its some type of race condition
	//
	// Test fix seems to confirm it. Now: keep it on cpu side or do a new cuda launch?
	/*
	if (x >= 5 && x*x <= in_n) {
		if (in_device_memory[x - 1]) {
			for (size_t y = x*x; y <= in_n; y += x*x) {
				in_device_memory[y - 1] = false;
			}
		}
	}
	*/
}

//Private------------------------------------------------------------------------------------------
void SieveAtkinCUDA::SieveKernel(unsigned int in_blocks, unsigned int in_threads, size_t in_start, size_t in_end, bool* in_mem_ptr) {
	AtkinKernel <<<in_blocks, in_threads, 0>>> (in_start, in_end, in_mem_ptr);
}

void SieveAtkinCUDA::DoSieve() {
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

	//Test Fix
	for (size_t x = 5; x*x <= this->end_; x++) {
		if (this->mem_class_ptr_->CheckIndex(x - 1)) {
			for (size_t y = x * x; y <= this->end_; y += x * x) {
				this->mem_class_ptr_->SetNonPrime(y - 1);
			}
		}
	}
	//Test Fix
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

	this->timer_.SaveTime();

	//Set 2 and 3 manually as sieving process starts at 5
	if (in_n >= 2) { this->mem_class_ptr_->SetPrime(1); }
	if (in_n >= 3) { this->mem_class_ptr_->SetPrime(2); }

	this->DoSieve();

	this->timer_.SaveTime();

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
