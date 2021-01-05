#ifndef SIEVE_ATKIN_CPU_H
#define SIEVE_ATKIN_CPU_H

#include "sieve_base.h"
#include "prime_memory/prime_memory_fragsafe.h"

class SieveAtkinCPU : public SieveBase {
private:
	void DoSieve();
	size_t IndexToNumber(size_t in_i);
public:
	SieveAtkinCPU(size_t in_n);
	SieveAtkinCPU(size_t in_n, PrimeMemoryFragsafe* in_ptr);
	~SieveAtkinCPU();

	bool IsPrime(size_t in_num);
};

#endif // !SIEVE_ATKIN_CPU_H