#ifndef PRIME_MEMORY_BIT_H
#define PRIME_MEMORY_BIT_H

#include "prime_memory.h"

class PrimeMemoryBit : public PrimeMemory {
private:
	struct mem_index {
		unsigned int arr_i = 0;
		unsigned int bit_i = 0;
	};

	unsigned int num_of_bits_ = 0;
	unsigned int num_of_ints_ = 0;
	int* bits_ = 0;

	mem_index AccessIndex(unsigned int in_i);

public:
	PrimeMemoryBit(unsigned int in_size);
	~PrimeMemoryBit();

	bool CheckIndex(unsigned int in_i);

	void SetNonPrime(unsigned int in_i);
	void SetPrime(unsigned int in_i);
};

#endif // !PRIME_MEMORY_BIT_H