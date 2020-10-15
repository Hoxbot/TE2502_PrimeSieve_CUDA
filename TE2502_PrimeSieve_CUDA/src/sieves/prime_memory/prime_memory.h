#ifndef PRIME_MEMORY_H
#define PRIME_MEMORY_H

class PrimeMemory {
private:

public:
	PrimeMemory() {};
	~PrimeMemory() {};

	virtual bool CheckIndex(unsigned int in_i) = 0;

	virtual void SetNonPrime(unsigned int in_i) = 0;
	virtual void SetPrime(unsigned int in_i) = 0;
};

#endif // !PRIME_MEMORY_H
