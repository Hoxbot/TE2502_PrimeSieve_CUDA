#ifndef PRIME_MEMORY_BOOL_H
#define PRIME_MEMORY_BOOL_H

#include "prime_memory.h"

class PrimeMemoryBool : public PrimeMemory {
protected:
	size_t mem_size_ = 0;
	bool* mem_arr_ = nullptr;

public:
	PrimeMemoryBool(size_t in_size);
	~PrimeMemoryBool();

	void* getMemPtr();
	virtual size_t BytesAllocated();
	virtual size_t NumberCapacity();

	bool CheckIndex(size_t in_i);

	void SetNonPrime(size_t in_i);
	void SetPrime(size_t in_i);
	void FlipPrime(size_t in_i);

	void SetAllNonPrime();
	void SetAllPrime();
};

#endif // !PRIME_MEMORY_BOOL_H
