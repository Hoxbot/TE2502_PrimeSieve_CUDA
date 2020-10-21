#include "prime_memory_bool.h"

//Private------------------------------------------------------------------------------------------


//Public-------------------------------------------------------------------------------------------
PrimeMemoryBool::PrimeMemoryBool(size_t in_size) {

	this->arr_size_ = in_size;
	this->tracker_arr_ = new bool[in_size]; //NTS: indexes [0, in_size-1]

	//Start all values as true ("known as primes")
	for (unsigned int i = 0; i < in_size; i++) {
		this->SetPrime(i);
	}
}

PrimeMemoryBool::~PrimeMemoryBool() {
	delete[] this->tracker_arr_;
	this->tracker_arr_ = nullptr;
}

bool PrimeMemoryBool::CheckIndex(size_t in_i) {
	return this->tracker_arr_[in_i];
}

void PrimeMemoryBool::SetNonPrime(size_t in_i) {
	this->tracker_arr_[in_i] = false;
}

void PrimeMemoryBool::SetPrime(size_t in_i) {
	this->tracker_arr_[in_i] = true;
}

size_t PrimeMemoryBool::BytesAllocated() {
	return this->arr_size_ * sizeof(bool);
}
