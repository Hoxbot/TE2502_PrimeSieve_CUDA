#include "sieve_sundaram_cpu.h"

//Private------------------------------------------------------------------------------------------
void SieveSundaramCPU::DoSieve() {
	
	size_t n = this->mem_class_ptr_->NumberCapacity();

	for (size_t i = this->start_; i <= n; i++) {
		//De-list all numbers that fullful the condition: (i + j + 2*i*j) <= n
		for (size_t j = i; (i + j + 2 * i*j) <=  n; j++) {
			this->mem_class_ptr_->SetNonPrime((i + j + 2*i*j) - this->start_);
		}
	}
}

size_t SieveSundaramCPU::IndexToNumber(size_t in_i) {
	return 2 * (in_i + this->start_) + 1;
}

//Public-------------------------------------------------------------------------------------------
SieveSundaramCPU::SieveSundaramCPU(size_t in_n)// {
	: SieveBase(1, in_n) {

	//Determine memory capacity needed
	size_t mem_size = ((in_n - 2) / 2) + ((in_n - 2) % 2);

	this->mem_class_ptr_ = new PrimeMemoryBool(mem_size);
	//this->mem_class_ptr_ = new PrimeMemoryBit(mem_size);

	//Sundaram starts all as primes
	this->mem_class_ptr_->SetAllPrime();

	this->timer_.SaveTime();

	this->DoSieve();

	this->timer_.SaveTime();
}

SieveSundaramCPU::SieveSundaramCPU(size_t in_n, PrimeMemoryFragsafe * in_ptr)// {
	: SieveBase(1, in_n) {

	//Determine memory capacity needed
	size_t mem_size = ((in_n - 2) / 2) + ((in_n - 2) % 2);

	//Set fragsafe memory
	in_ptr->AllocateSubMemory(mem_size);
	this->mem_class_ptr_ = in_ptr;

	//Sundaram starts all as primes
	this->mem_class_ptr_->SetAllPrime();

	this->timer_.SaveTime();

	this->DoSieve();

	this->timer_.SaveTime();
}

SieveSundaramCPU::~SieveSundaramCPU() {
	//Do not delete memory if its a fragsafe pointer
	if (dynamic_cast<PrimeMemoryFragsafe*>(this->mem_class_ptr_) != nullptr) { return; }

	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}
}

bool SieveSundaramCPU::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }

	//Anything smaller than 2 is not a prime
	if (in_num < 2) { return false; }

	//Sundaram's sieve does not store even numbers
	//> 2 special case
	//> All other even numbers false
	if (in_num == 2) { return true; }
	if ((in_num % 2) == 0) { return false; }

	//For odd numbers, offset number to correct index
	size_t the_number_index = ((in_num - 1) / 2) - this->start_;

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}
