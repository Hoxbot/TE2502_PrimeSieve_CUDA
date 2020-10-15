#include "prime_memory_bool.h"

//Private------------------------------------------------------------------------------------------


//Public-------------------------------------------------------------------------------------------
PrimeMemoryBool::PrimeMemoryBool(unsigned int in_size) {
	this->tracker_arr_ = new bool[in_size]; //NTS: indexes [0, in_size-1]

	//Start all values as true ("known as primes")
	for (unsigned int i = 0; i < in_size; i++) {
		this->SetPrime(i);
	}
}

PrimeMemoryBool::~PrimeMemoryBool() {
	delete[] this->tracker_arr_;
}

bool PrimeMemoryBool::CheckIndex(unsigned int in_i) {
	return this->tracker_arr_[in_i];
}

void PrimeMemoryBool::SetNonPrime(unsigned int in_i) {
	this->tracker_arr_[in_i] = false;
}

void PrimeMemoryBool::SetPrime(unsigned int in_i) {
	this->tracker_arr_[in_i] = true;
}