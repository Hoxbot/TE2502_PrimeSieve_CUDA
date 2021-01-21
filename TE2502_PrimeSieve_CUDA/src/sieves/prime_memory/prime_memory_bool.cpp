#include "prime_memory_bool.h"

#include <cstring>

//Private------------------------------------------------------------------------------------------


//Public-------------------------------------------------------------------------------------------
PrimeMemoryBool::PrimeMemoryBool(size_t in_size) {

	this->mem_size_ = in_size;
	this->mem_arr_ = new bool[in_size]; //NTS: indexes [0, in_size-1]
}

PrimeMemoryBool::~PrimeMemoryBool() {
	delete[] this->mem_arr_;
	this->mem_arr_ = nullptr;
}

void* PrimeMemoryBool::getMemPtr() {
	return this->mem_arr_;
}

size_t PrimeMemoryBool::BytesAllocated() {
	return this->mem_size_ * sizeof(bool);
}

size_t PrimeMemoryBool::NumberCapacity() {
	return this->mem_size_;
}

bool PrimeMemoryBool::CheckIndex(size_t in_i) {
	return this->mem_arr_[in_i];
}

void PrimeMemoryBool::SetNonPrime(size_t in_i) {
	this->mem_arr_[in_i] = false;
}

void PrimeMemoryBool::SetPrime(size_t in_i) {
	this->mem_arr_[in_i] = true;
}

void PrimeMemoryBool::FlipPrime(size_t in_i) {
	this->mem_arr_[in_i] = !this->mem_arr_[in_i];
}

void PrimeMemoryBool::SetAllNonPrime() {
	/*
	for (unsigned int i = 0; i < this->mem_size_; i++) {
		this->SetNonPrime(i);
	}
	*/

	std::memset(this->mem_arr_, false, this->mem_size_);
}

void PrimeMemoryBool::SetAllPrime() {
	/*
	for (unsigned int i = 0; i < this->mem_size_; i++) {
		this->SetPrime(i);
	}
	*/

	std::memset(this->mem_arr_, true, this->mem_size_);
}


