#include "sieve_sundaram_cuda.cuh"

//CUDA---------------------------------------------------------------------------------------------


//Private------------------------------------------------------------------------------------------
void SieveSundaramCUDA::UploadMemory() {
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


	//Step 0: 
	//NTS: Do we benefit from contious allocation?


	//Step 1: Allocate memory on device
	size_t num_of_bytes = this->mem_class_ptr_->BytesAllocated();
	CUDAErrorOutput(
		cudaMalloc((void**)&(this->device_mem_ptr_), num_of_bytes),
		"cudaMalloc()", __FUNCTION__
	);

	//Step 2: Copy data to memory
	//NTS: The booleans must be true, but is an upload needed?
	//Is it enough to allocate, alter in the kernel and download?
	CUDAErrorOutput(
		cudaMemcpy(
			this->device_mem_ptr_,			//Target
			/*TBA*/,						//Source	//Working here
			num_of_bytes,					//Byte count
			cudaMemcpyHostToDevice			//Transfer type
		),
		"cudaMemcpy()", __FUNCTION__
	);
}

void SieveSundaramCUDA::DownloadMemory() {
	//WIP
}

void SieveSundaramCUDA::DoSieve() {

	//WIP

}

size_t SieveSundaramCUDA::IndexToNumber(size_t in_i) {
	return this->start_ + in_i;
}

//Public-------------------------------------------------------------------------------------------
SieveSundaramCUDA::SieveSundaramCUDA(size_t in_n)// {
	: SieveBase(1, in_n) {

	//this->mem_class_ptr_ = new PrimeMemoryBool(this->n_);
	this->mem_class_ptr_ = new PrimeMemoryBit(this->n_);

	this->private_timer_.SaveTime();

	this->DoSieve();

	this->private_timer_.SaveTime();

}

SieveSundaramCUDA::~SieveSundaramCUDA() {
	if (this->mem_class_ptr != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}	
}

bool SieveSundaramCUDA::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	//Otherwise return the stored bool for that value

	//Offset number to correct index
	size_t the_number_index = in_num - this->start_;

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}
