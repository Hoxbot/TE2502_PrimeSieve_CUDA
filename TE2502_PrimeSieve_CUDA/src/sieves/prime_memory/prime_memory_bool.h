#ifndef PRIME_MEMORY_BOOL_H
#define PRIME_MEMORY_BOOL_H

#include "prime_memory.h"

class PrimeMemoryBool : public PrimeMemory {
private:
	bool* tracker_arr_ = nullptr;

public:
	PrimeMemoryBool(size_t in_size);
	~PrimeMemoryBool();

	bool CheckIndex(size_t in_i);

	void SetNonPrime(size_t in_i);
	void SetPrime(size_t in_i);
};

#endif // !PRIME_MEMORY_BOOL_H
