#include "sieve_eratosthenes_cpu.h"

//#include <iostream>

//Private------------------------------------------------------------------------------------------
void SieveErathosthenesCPU::DoSieve() {
	unsigned int root_of_end = std::sqrt(this->end_) + 1;	//NTS:	As noted in the following comment, all composite numbers
															//		lower than i^2 for a given i has already been calculated
															//		for previous (lower) i:s.
															//		This means we start each iteration with i^2,  which means
															//		we need not go further than the root of n, since:
															//			if (i > root(n)) => i^2 > n, and n is max

	for (size_t i = this->start_; i < root_of_end; i++) {
		if (this->mem_class_ptr_->CheckIndex(i - this->start_)) {
			for (size_t j = i * i; j <= this->end_; j = j + i) {	//NTS:	Start value is i^2.
																		//		This because all composites lower than i^2
																		//		will have been covered by lower i:s

				//std::cout << "\tSieve Set NonPrime:\t" << j << "\n";

				this->mem_class_ptr_->SetNonPrime(j - this->start_);
			}
		}
	}
}

size_t SieveErathosthenesCPU::IndexToNumber(size_t in_i) {
	return this->start_ + in_i;
}


//Public-------------------------------------------------------------------------------------------
SieveErathosthenesCPU::SieveErathosthenesCPU(size_t in_n)// {
	: SieveBase(2, in_n) {

	//this->mem_class_ptr_ = new PrimeMemoryBool(this->n_);
	this->mem_class_ptr_ = new PrimeMemoryBit(this->n_);

	this->private_timer_.SaveTime();

	this->DoSieve();

	this->private_timer_.SaveTime();

}

SieveErathosthenesCPU::~SieveErathosthenesCPU() {
	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}
}

bool SieveErathosthenesCPU::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	//Otherwise return the stored bool for that value

	//Offset number to correct index
	size_t the_number_index = in_num - this->start_;

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}