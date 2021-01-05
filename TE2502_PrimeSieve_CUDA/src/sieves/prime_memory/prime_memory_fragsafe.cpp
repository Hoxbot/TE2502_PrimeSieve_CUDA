#include "prime_memory_fragsafe.h"

//Private------------------------------------------------------------------------------------------




//Public-------------------------------------------------------------------------------------------
PrimeMemoryFragsafe::PrimeMemoryFragsafe(size_t in_size)
	: PrimeMemoryBool(in_size) {

	this->allocated_size_ = this->mem_size_;
}

PrimeMemoryFragsafe::~PrimeMemoryFragsafe() {
	//Calls parent destructor
}

bool PrimeMemoryFragsafe::AllocateSubMemory(size_t in_alloc_size) {
	//If possible, "allocates" memory within the existing allocation
	//After that point, this prime memory masquerades as a PrimeMemoryBool
	//of size 'in_alloc_size'

	if (in_alloc_size > this->mem_size_) { return false; }

	this->allocated_size_ = in_alloc_size;

	return true;
}

/*
void* PrimeMemoryFragsafe::getMemPtr() {
	return this->mem_arr_;
}
*/

size_t PrimeMemoryFragsafe::BytesAllocated() {
	return this->allocated_size_ * sizeof(bool);
}

size_t PrimeMemoryFragsafe::NumberCapacity() {
	return allocated_size_;
}

/*
bool PrimeMemoryFragsafe::CheckIndex(size_t in_i) {
	return this->mem_arr_[in_i];
}

void PrimeMemoryFragsafe::SetNonPrime(size_t in_i) {
	this->mem_arr_[in_i] = false;
}

void PrimeMemoryFragsafe::SetPrime(size_t in_i) {
	this->mem_arr_[in_i] = true;
}

void PrimeMemoryFragsafe::FlipPrime(size_t in_i) {
	this->mem_arr_[in_i] = !this->mem_arr_[in_i];
}

void PrimeMemoryFragsafe::SetAllNonPrime() {
	for (unsigned int i = 0; i < this->allocated_size_; i++) {
		this->SetNonPrime(i);
	}
}

void PrimeMemoryFragsafe::SetAllPrime() {
	for (unsigned int i = 0; i < this->allocated_size_; i++) {
		this->SetPrime(i);
	}
}
*/