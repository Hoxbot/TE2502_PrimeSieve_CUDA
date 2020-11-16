#include "sieve_atkin_cpu.h"

//#include <iostream>

//Private------------------------------------------------------------------------------------------
void SieveAtkinCPU::DoSieve() {

	//Sieve of Atkins
	//> For (x^2 <= n) and (y^2 <= n), x = 1,2,..., y = 1,2,...
	//> A number is prime if any of the following is true:
	//>> (z = 4*x*x + y*y) has odd number of solutions	AND	(z % 12 = 1) or (z % 12 = 5)
	//>> (z = 3*x*x + y*y) has odd number of solutions	AND	(z % 12 = 7)
	//>> (z = 3*x*x - y*y) has odd number of solutions	AND (x > y)	AND	(z % 12 = 11)
	//> Multiples of squares might have been marked, delist:
	//>> (z = x*x*y), x = 1,2,..., y = 1,2,...

	//NTS:	"Odd number of solutions", does that mean we should flip the state to the inverse?
	//		Two hits (even number of solutions) would then flip false->true->false
	//Ans:	Yes, apparently.

	for (size_t x = 1; x*x <= this->end_; x++) {
		for (size_t y = 1; y*y <= this->end_; y++) {

			size_t z = (4 * x*x) + (y*y);
			if (z <= this->end_ && (z % 12 == 1 || z % 12 == 5)) { this->mem_class_ptr_->FlipPrime(z - 1); }

			z = (3 * x*x) + (y*y);
			if (z <= this->end_ && (z % 12 == 7)) { this->mem_class_ptr_->FlipPrime(z - 1); }

			z = (3 * x*x) - (y*y);
			if (z <= this->end_ && (x > y) && (z % 12 == 11)) { this->mem_class_ptr_->FlipPrime(z - 1); }
		}
	}

	for (size_t x = 5; x*x <= this->end_; x++) {
		if (this->mem_class_ptr_->CheckIndex(x - 1)) {
			for (size_t y = x * x; y <= this->end_; y += x*x) {
				this->mem_class_ptr_->SetNonPrime(y - 1);
			}
		}
	}
}

size_t SieveAtkinCPU::IndexToNumber(size_t in_i) {
	return this->start_ + in_i;
}


//Public-------------------------------------------------------------------------------------------
SieveAtkinCPU::SieveAtkinCPU(size_t in_n)// {
	: SieveBase(1, in_n) {

	//NTS: Atkins excluding limit? ( [1, n[ )

	//Determine memory capacity needed
	size_t mem_size = in_n;

	this->mem_class_ptr_ = new PrimeMemoryBool(mem_size);
	//this->mem_class_ptr_ = new PrimeMemoryBit(mem_size);

	//Atkin starts all as non-primes
	this->mem_class_ptr_->SetAllNonPrime();

	this->timer_.SaveTime();

	//Set 2 and 3 manually as sieving process starts at 5
	if (in_n >= 2) { this->mem_class_ptr_->SetPrime(1); }
	if (in_n >= 3) { this->mem_class_ptr_->SetPrime(2); }

	this->DoSieve();

	this->timer_.SaveTime();

}

SieveAtkinCPU::~SieveAtkinCPU() {
	if (this->mem_class_ptr_ != nullptr) {
		delete this->mem_class_ptr_;
		this->mem_class_ptr_ = nullptr;
	}
}

bool SieveAtkinCPU::IsPrime(size_t in_num) {
	//Everything outside scope is false
	if (in_num < this->start_ || in_num > this->end_) { return false; }
	//Otherwise return the stored bool for that value

	//Offset number to correct index
	size_t the_number_index = in_num - this->start_;

	//Return
	return this->mem_class_ptr_->CheckIndex(the_number_index);
}