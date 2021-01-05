#ifndef PRIME_MEMORY_FRAGSAFE_H
#define PRIME_MEMORY_FRAGSAFE_H

#include "prime_memory_bool.h"

class PrimeMemoryFragsafe : public PrimeMemoryBool {
private:
	//size_t mem_size_ = 0;
	//bool* mem_arr_ = nullptr;

	size_t allocated_size_ = 0;

public:
	PrimeMemoryFragsafe(size_t in_total_size);
	~PrimeMemoryFragsafe();

	bool AllocateSubMemory(size_t in_alloc_size);

	//void* getMemPtr();
	size_t BytesAllocated();
	size_t NumberCapacity();

	/*
	bool CheckIndex(size_t in_i);

	void SetNonPrime(size_t in_i);
	void SetPrime(size_t in_i);
	void FlipPrime(size_t in_i);

	void SetAllNonPrime();
	void SetAllPrime();
	*/

};

#endif // !PRIME_MEMORY_FRAGSAFE_H
