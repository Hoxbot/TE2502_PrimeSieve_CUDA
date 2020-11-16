#include "prime_memory_bool.h"

//Private------------------------------------------------------------------------------------------


//Public-------------------------------------------------------------------------------------------
PrimeMemoryBool::PrimeMemoryBool(size_t in_size) {

	this->arr_size_ = in_size;
	this->tracker_arr_ = new bool[in_size]; //NTS: indexes [0, in_size-1]
}

PrimeMemoryBool::~PrimeMemoryBool() {
	delete[] this->tracker_arr_;
	this->tracker_arr_ = nullptr;
}

void* PrimeMemoryBool::getMemPtr() {
	return this->tracker_arr_;
}

size_t PrimeMemoryBool::BytesAllocated() {
	return this->arr_size_ * sizeof(bool);
}

size_t PrimeMemoryBool::NumberCapacity() {
	return this->arr_size_;
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

void PrimeMemoryBool::FlipPrime(size_t in_i) {
	this->tracker_arr_[in_i] = !this->tracker_arr_[in_i];
}

void PrimeMemoryBool::SetAllNonPrime() {
	for (unsigned int i = 0; i < this->arr_size_; i++) {
		this->SetNonPrime(i);
	}
}

void PrimeMemoryBool::SetAllPrime() {
	for (unsigned int i = 0; i < this->arr_size_; i++) {
		this->SetPrime(i);
	}
}


