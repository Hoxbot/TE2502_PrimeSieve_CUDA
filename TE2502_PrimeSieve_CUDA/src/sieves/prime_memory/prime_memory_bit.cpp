#include "prime_memory_bit.h"

//Private------------------------------------------------------------------------------------------
PrimeMemoryBit::mem_index PrimeMemoryBit::AccessIndex(unsigned int in_i) {
	//Create mem_index
	mem_index ret_i;

	//Determine array index and bit index
	ret_i.arr_i = in_i / 32;
	ret_i.bit_i = in_i % 32;

	//Return
	return ret_i;
}

//Public-------------------------------------------------------------------------------------------
PrimeMemoryBit::PrimeMemoryBit(unsigned int in_size) {
	
	//Store the number of intended bits
	this->num_of_bits_ = in_size;

	//Each int can hold 32 bits (1 int = 4 bytes = 32 bits)
	//+1 to int array size to hold the amount of bits that 
	//do not fill a full int 
	this->num_of_ints_ = (in_size / 32) + 1;

	//Allocate space
	this->bits_ = new int[this->num_of_ints_];

	//Start all values as true ("known as primes")
	for (unsigned int i = 0; i < in_size; i++) {
		this->SetPrime(i);
	}
}

PrimeMemoryBit::~PrimeMemoryBit() {
	delete[] this->bits_;
	this->bits_ = nullptr;
}

bool PrimeMemoryBit::CheckIndex(unsigned int in_i) {
	//Return false for anything outside intended storage
	//if (in_i >= this->num_of_bits_) { return false; }

	//Calc array index and bit index
	mem_index m = this->AccessIndex(in_i);

	// Create an integer 1		:	0000000...00001
	// Shift it by bit_i bits	:	00000...1...000
	// Mask the arr_i position	:	1 or 0
	bool ret_bit = this->bits_[m.arr_i] & ((int)1 << m.bit_i);

	//Return
	return ret_bit;
}

void PrimeMemoryBit::SetNonPrime(unsigned int in_i) {
	//Return directly for anything outside intended storage
	//if (in_i >= this->num_of_bits_) { return false; }

	//Set bit to true (|= is bitwise inclusive OR)
	mem_index m = this->AccessIndex(in_i);
	this->bits_[m.arr_i] |= true << m.bit_i;
}

void PrimeMemoryBit::SetPrime(unsigned int in_i) {
	//Return directly for anything outside intended storage
	//if (in_i >= this->num_of_bits_) { return false; }

	//Set bit to false (|= is bitwise inclusive OR)
	mem_index m = this->AccessIndex(in_i);
	this->bits_[m.arr_i] |= false << m.bit_i;
}
