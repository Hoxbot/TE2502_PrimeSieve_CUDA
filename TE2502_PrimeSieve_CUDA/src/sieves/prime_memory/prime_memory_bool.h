#ifndef PRIME_MEMORY_BOOL_H
#define PRIME_MEMORY_BOOL_H

#include "prime_memory.h"

class PrimeMemoryBool : public PrimeMemory {
private:
	size_t arr_size_ = 0;
	bool* tracker_arr_ = nullptr;

public:
	PrimeMemoryBool(size_t in_size);
	~PrimeMemoryBool();

	void* getMemPtr();
	size_t BytesAllocated();
	size_t NumberCapacity();

	bool CheckIndex(size_t in_i);

	void SetNonPrime(size_t in_i);
	void SetPrime(size_t in_i);

	void SetAllNonPrime();
	void SetAllPrime();
};

#endif // !PRIME_MEMORY_BOOL_H
