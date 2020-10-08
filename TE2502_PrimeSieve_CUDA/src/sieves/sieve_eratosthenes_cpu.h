#ifndef SIEVE_ERATOSTHENES_CPU_H
#define SIEVE_ERATOSTHENES_CPU_H

#include "sieve_base.h"

class SieveErathosthenesCPU : public SieveBase {
private:
	void DoSieve();
public:
	SieveErathosthenesCPU(unsigned int in_n);
	~SieveErathosthenesCPU();
};

#endif // !SIEVE_ERATOSTHENES_CPU_H