#ifndef PRIME_MEMORY_H
#define PRIME_MEMORY_H

class PrimeMemory {
private:

public:
	PrimeMemory() {};
	virtual ~PrimeMemory() {};

	virtual bool CheckIndex(size_t in_i) = 0;

	virtual void SetNonPrime(size_t in_i) = 0;
	virtual void SetPrime(size_t in_i) = 0;
};

#endif // !PRIME_MEMORY_H
