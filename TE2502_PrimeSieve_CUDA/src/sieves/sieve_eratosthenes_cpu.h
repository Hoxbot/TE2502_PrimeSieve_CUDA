#ifndef SIEVE_ERATOSTHENES_CPU_H
#define SIEVE_ERATOSTHENES_CPU_H

#include "sieve_base.h"
#include "prime_memory/prime_memory_fragsafe.h"

class SieveEratosthenesCPU : public SieveBase {
private:
	//size_t start_ = 0;
	//size_t end_ = 0;
	//unsigned int step_length_ = 1;

	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveEratosthenesCPU(size_t in_n);
	SieveEratosthenesCPU(size_t in_n, PrimeMemoryFragsafe* in_ptr);
	~SieveEratosthenesCPU();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_ERATOSTHENES_CPU_H