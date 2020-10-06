#include "sieve_base.h"

//Private------------------------------------------------------------------------------------------
void SieveBase::flip(in_index) {
	this->tracker_arr_[in_index] = !this->tracker_arr_[in_index];
}

void SieveBase::SimpleEratosthenes() {
	unsigned int root_of_n = 0;
	for (unsigned int i = 2; i < root_of_n; i++) {
		if (this->tracker_arr_[i]) {
			for (unsigned int j = 2 * i; j <= this->n_; j + i) {	//NTS: Start value i^2?
				this->flip(j)
			}
		}
	}
}


//Public-------------------------------------------------------------------------------------------
SieveBase::SieveBase(unsigned int in_n) {
	this->n_ = in_n;
	this->tracker_arr_ = new bool[n_]; //NTS: Includes '1' currently 
	//Start all values as true ("known as primes")
	for (unsigned int i = 0; i < this->n_; i++) {
		this->tracker_arr_[i] = true;
	}
}

SieveBase::~SieveBase() {
	delete[] this->tracker_arr_;
}
