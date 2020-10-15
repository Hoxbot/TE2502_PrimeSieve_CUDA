#include "sieve_eratosthenes_cpu.h"

//Private------------------------------------------------------------------------------------------
void SieveErathosthenesCPU::DoSieve() {
	unsigned int root_of_n = std::sqrt(this->n_) + 1;	//NTS:	As noted in the following comment, all composite numbers
														//		lower than i^2 for a given i has already been calculated
														//		for previous (lower) i:s.
														//		This means we start each iteration with i^2,  which means
														//		we need not go further than the root of n, since:
														//			if (i > root(n)) => i^2 > n, and n is max

	for (unsigned int i = 2; i < root_of_n; i++) {
		if (this->mem_class_ptr_->CheckIndex(i-2)) {
			for (unsigned int j = i * i; j <= this->n_; j = j + i) {	//NTS:	Start value is i^2.
																	//		This because all composites lower than i^2
																	//		will have been covered by lower i:s
				this->mem_class_ptr_->SetNonPrime(j-2);
			}
		}
	}
}


//Public-------------------------------------------------------------------------------------------
SieveErathosthenesCPU::SieveErathosthenesCPU(unsigned int in_n) {
//	: SieveBase(2, in_n) {

	//WIP
	//Calc start number
	//Calc array size dependant on start, max and step length

	this->start_ = 2;
	this->end_ = in_n;

	this->n_ = this->end_ - this->start_;
	//WIP

	this->mem_class_ptr_ = new PrimeMemoryBool(this->n_);

	this->private_timer_.SaveTime();

	this->DoSieve();

	this->private_timer_.SaveTime();

}

SieveErathosthenesCPU::~SieveErathosthenesCPU() {
	//Calls base destructor on auto
}

bool SieveErathosthenesCPU::IsPrime(unsigned int in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	//Otherwise return the stored bool for that value

	//WIP
	unsigned int the_number_index = in_num - this->start_;
	//WIP

	return this->mem_class_ptr_->CheckIndex(the_number_index);
}