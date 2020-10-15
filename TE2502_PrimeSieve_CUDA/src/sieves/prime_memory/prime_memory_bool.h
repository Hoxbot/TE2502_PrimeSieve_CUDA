#ifndef PRIME_MEMORY_BOOL_H
#define PRIME_MEMORY_BOOL_H

#include "prime_memory.h"

class PrimeMemoryBool : public PrimeMemory {
private:
	bool* tracker_arr_ = nullptr;

public:
	PrimeMemoryBool(unsigned int in_size);
	~PrimeMemoryBool();

	bool CheckIndex(unsigned int in_i);

	void SetNonPrime(unsigned int in_i);
	void SetPrime(unsigned int in_i);
};

#endif // !PRIME_MEMORY_BOOL_H
