#ifndef PRIME_MEMORY_H
#define PRIME_MEMORY_H

class PrimeMemory {
private:

public:
	PrimeMemory() {};
	virtual ~PrimeMemory() {};

	virtual void* getMemPtr() = 0;
	virtual size_t BytesAllocated() = 0;
	virtual size_t NumberCapacity() = 0;

	virtual bool CheckIndex(size_t in_i) = 0;

	virtual void SetNonPrime(size_t in_i) = 0;
	virtual void SetPrime(size_t in_i) = 0;
	virtual void FlipPrime(size_t in_i) = 0;

	virtual void SetAllNonPrime() = 0;
	virtual void SetAllPrime() = 0;

	
};

#endif // !PRIME_MEMORY_H
