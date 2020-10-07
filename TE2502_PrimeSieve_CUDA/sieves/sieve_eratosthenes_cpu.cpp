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
		if (this->tracker_arr_[i]) {
			for (unsigned int j = i * i; j < this->n_; j = j + i) {	//NTS:	Start value is i^2.
																	//		This because all composites lower than i^2
																	//		will have been covered by lower i:s
				this->SetNonPrime(j);
			}
		}
	}
}


//Public-------------------------------------------------------------------------------------------
SieveErathosthenesCPU::SieveErathosthenesCPU(unsigned int in_n) 
	: SieveBase(in_n) {

	this->private_timer_.SaveTime();

	this->DoSieve();

	this->private_timer_.SaveTime();

}

SieveErathosthenesCPU::~SieveErathosthenesCPU() {
	//Calls base destructor on auto
}
