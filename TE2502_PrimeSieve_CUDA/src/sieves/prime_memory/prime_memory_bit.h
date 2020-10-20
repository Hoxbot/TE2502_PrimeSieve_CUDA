#ifndef PRIME_MEMORY_BIT_H
#define PRIME_MEMORY_BIT_H

#include "prime_memory.h"

class PrimeMemoryBit : public PrimeMemory {
private:
	struct mem_index {
		size_t arr_i = 0;
		size_t bit_i = 0;
	};

	size_t num_of_bits_ = 0;
	size_t num_of_ints_ = 0;
	int* bits_ = 0;

	mem_index AccessIndex(size_t in_i);

public:
	PrimeMemoryBit(size_t in_size);
	~PrimeMemoryBit();

	bool CheckIndex(size_t in_i);

	void SetNonPrime(size_t in_i);
	void SetPrime(size_t in_i);
};

#endif // !PRIME_MEMORY_BIT_H